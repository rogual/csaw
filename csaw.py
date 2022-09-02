#!/usr/bin/env python3

"""

CSAW by Robin Allen <r@foon.uk>


Licence
=======

You may redistribute csaw as long as the original author remains
acknowledged as above, and this licence text is included
unmodified.

If you use csaw to build your software, an acknowledgement in
the product documentation would be appreciated but is not
required.


TODO
====

  namespace
    Splitting a namespace over several files isn't supported
    yet. Shouldn't be too hard, I just never use it.

  template class
    could be smarter and actually split out member fndefs
    probably no point for now

  template member function
    definition should go in header, not source

  R"" strings and other c++11 weirdo strings

  C++11 {}-initialization e.g. int a{4};

Areas where csaw needs some help
================================

  Nested classes
    Use #pragma depends Parent if you use Parent::x

  Return nested type
    Always fully-qualify return types, e.g. 
    class C { struct T {}; C::T f() {} };

  Templates
    No way of knowing whether template params need to be fully
    defined or not. We just assume they do.

    Idea: Maybe introduce something like
        #pragma depends<T> vector<T> T

Preprocessor
============

   Not supported. Preprocess your files before csaw sees them. One
   custom directive is supported, #pragma depends, which marks all items
   in a file as depending on a specific name.

Qt
==

    Pass -qt and Q_OBJECT will be recognized. This makes the emitted
    source file suitable for processing with moc.
"""

from collections import defaultdict

import concurrent.futures
import contextlib
import threading
import functools
import argparse
import string
import array
import sys
import re
import os


line_directives_enabled = True
qt_enabled = False


TComment = sys.intern('Comment')
TString = sys.intern('String')
TWord = sys.intern('Word')
TPunctuation = sys.intern('Punctuation')
TNumber = sys.intern('Number')
TDirective = sys.intern('Directive')
TEnd = sys.intern('End')

local = threading.local()
    
local.RecordScope = []


def fatal(msg):
    print('\n--- FATAL ERROR ---', file=sys.stderr)
    for item in local.RecordScope:
        print('While parsing ', item, file=sys.stderr)
        print(msg, file=sys.stderr)
    raise Exception(msg)
    sys.exit(1)


@contextlib.contextmanager
def to_file(path_or_file, mode):
    if isinstance(path_or_file, str):
        with open(path_or_file, mode) as f:
            yield f
    else:
        yield path_or_file


@contextlib.contextmanager
def append(xs, x):
    xs.append(x)
    yield
    xs.pop()


class Token:
    def __init__(self, lexer, type_, index, length):
        self.lexer = lexer
        self.type = type_
        self.index = index
        self.length = length

        self.mapped_line_offset = 0
        self.mapped_path = None

    @property
    def text(self):
        if self.type == TEnd:
            return 'end of file'
        return self.lexer.source_text[self.index:self.index+self.length]

    @property
    def text_with_whitespace(self):
        if self.type == TEnd:
            return 'end of file'

        i = self.index
        j = self.index + self.length
        while j < len(self.lexer.source_text) and self.lexer.source_text[j].isspace():
            j += 1

        return self.lexer.source_text[i:j]

    @property
    def real_line(self):
        if self.type == TEnd:
            return self.lexer.line_map[-1] + 1
        return self.lexer.line_map[self.index]

    @property
    def line(self):
        return self.real_line + self.mapped_line_offset

    @property
    def path(self):
        return self.mapped_path or self.lexer.input_path

    @property
    def line_directive(self):
        return '#line %i "%s"\n' % (self.line, self.path.replace('\\', '\\\\'))
        
    def __repr__(self):
        return '%s "%s"' % (self.type, self.text)


class Lexer:
    regexes = [
        (None, re.compile('|'.join([
            r'\s+',
            r'//.*?(\n|$)',
            r'/\*(.|\n)*?\*/'
        ]))),
        (TWord, re.compile(r'@?[A-Za-z_][A-Za-z_0-9]*')),
        (TString, re.compile('|'.join([
            r'"([^"\\]|\\.)*"',
            r"'([^'\\]|\\.)*'"
        ]))),
        (TDirective, re.compile(r"^#[A-Za-z0-9_]+", re.MULTILINE)),
        (TPunctuation, re.compile('|'.join([
            r'<<',
            r'\[\[',
            r'\]\]',
            r'::',
            r'->',
            '[' + re.escape(string.punctuation) + ']'
        ]))),
        (TNumber, re.compile(r'[0-9][0-9A-Fa-f.]*')),
    ]

    line_directive_regex = re.compile('(\d+) "(.*)"')

    @functools.cached_property
    def line_map(self):
        source_length = len(self.source_text)
        line_map = array.array('L', [0]) * source_length
        line = 1
        for i, char in enumerate(self.source_text):
            if char == '\n':
                line += 1
            line_map[i] = line
        return line_map

    def __init__(self, input_path):
        if isinstance(input_path, str):
            self.input_path = input_path
            with open(input_path, 'rt') as f:
                self.source_text = f.read()
        else:
            self.input_path = '<memory>'
            self.source_text = input_path.read()

        source_length = len(self.source_text)

        self.tokens = []

        # Tokenize
        mapped_path = None
        mapped_line_offset = 0
        
        pos = 0
        while pos < source_length:
            for type_, regex in Lexer.regexes:
                match = regex.match(self.source_text, pos)
                if match:
                    span = match.span(0)
                    length = span[1] - span[0]
                    assert length > 0
                    token = None

                    if type_ is not None:
                        token = Token(self, type_, pos, length)

                        if mapped_path is not None:
                            token.mapped_path = mapped_path
                            token.mapped_line_offset = mapped_line_offset

                    # Handle line directives
                    if token and (token.text == '#' or token.text == '#line'):

                        line_end = self.source_text.find('\n', pos)

                        if self.source_text[pos + len(token.text)] == ' ':
                            m = Lexer.line_directive_regex.search(
                                self.source_text,
                                pos,
                                line_end
                            )

                            if m:
                                mapped_line = int(m.group(1))
                                mapped_path = m.group(2)
                                mapped_line_offset = mapped_line - token.real_line - 1

                            else:
                                fatal("Wonky line directive at index %i of '%s'" % (pos, input_path))

                            pos = line_end

                        else:
                            pos += length

                    # Normal token
                    else:
                        if token:
                            self.tokens.append(token)
                        pos += length

                    break
            else:
                fatal('Unexpected character at index %i of "%s"' % (pos, input_path))

        self.tokens.append(Token(self, TEnd, source_length, 0))

    def __iter__(self):
        return iter(self.tokens)

    def __getitem__(self, i):
        return self.tokens[i]


class ParseError(Exception):
    pass


class Cursor:
    def __init__(self, tokens, index):
        self.tokens = tokens
        self.index = index

    def __bool__(self):
        return self.token.type != TEnd

    def next(self):
        if self.token.type == TEnd:
            raise StopIteration
        self.index += 1

    def error(self, msg):
        e = ParseError(
            '%s:%i:%s' % (self.token.path, self.token.line, msg)
        )
        e.record_scope = local.RecordScope[:]
        raise e

    @property
    def token(self):
        return self.tokens[self.index]

    @property
    def text(self):
        return self.token.text

    @property
    def type(self):
        return self.token.type

    def copy(self):
        return Cursor(self.tokens, self.index)

    def set(self, other):
        self.tokens = other.tokens
        self.index = other.index


class TokenRange:
    def __init__(self, start_cursor, end_cursor=None):
        self.start_cursor = start_cursor.copy()
        if end_cursor:
            self.set_end(end_cursor)
        else:
            self.end_cursor = None

    def set_end(self, end_cursor):
        self.end_cursor = end_cursor.copy()
        self.end_cursor.index -= 1

    @property
    def text(self):

        tokens = self.start_cursor.tokens

        if not self.end_cursor:
            return tokens.source_text[self.start_cursor.token.index:]

        i = self.start_cursor.token.index
        j = self.end_cursor.token.index + len(self.end_cursor.text)

        return tokens.source_text[i:j]

    @property
    def tokens(self):
        return self.start_cursor.tokens[
            self.start_cursor.index :
            self.end_cursor.index + 1
        ]

    @property
    def line_directive(self):
        return self.start_cursor.token.line_directive

    def emit_line_directive(self, f):
        if line_directives_enabled:
            f.write(self.line_directive)


class Node:

    dump_text = False
    
    @property
    def text(self):
        return self.range.text

    @property
    def tokens(self):
        return self.range.tokens

    @property
    def line_directive(self):
        return self.range.line_directive

    def emit_line_directive(self, f):
        if line_directives_enabled:
            f.write(self.line_directive)

    def __repr__(self):
        r = '%s(%s)' % (
            self.__class__.__name__,
            ', '.join(
                '%s=%s' % (k, v)
                for k, v in self.__dict__.items()
                if type(v) is str
                or v is True
            )
        )
        if self.dump_text:
            r += ' <"' + self.text + '">'
        return r

    def dump(self, indent=0):
        tab = '  ' * indent
        if indent == 0:
            print(repr(self))
        for k, v in self.__dict__.items():
            if type(v) is list:
                for i, item in enumerate(v):
                    if isinstance(item, Node):
                        print('%s  %s[%s]: %s' % (tab, k, i, repr(item)))
                        item.dump(indent+1)
            elif isinstance(v, Node):
                print('%s  %s: %s' % (tab, k, repr(v)))
                v.dump(indent+1)

    @classmethod
    def parse(cls, cursor):
        start_cursor = cursor.copy()
        node = cls._parse(cursor)
        if node:
            node.range = TokenRange(start_cursor, cursor)
            return node

    def emit_forward_declaration(self, f):
        pass

    def emit_interface(self, f):
        pass

    def emit_implementation(self, f):
        pass

    def emit_inline_function_definitions(self, f):
        pass

    def get_dependencies(self, names, typedef_names):
        return set()


class BaseRecord(Node):

    @classmethod
    def _parse(cls, cursor):

        self = BaseRecord()
        self.access = None
        self.name = None
        self.template_params = None
        self.is_virtual = False

        while cursor:
            if cursor.text in ['public', 'protected', 'private']:
                if self.access:
                    cursor.error('Too many access specifiers')
                self.access = cursor.text
                cursor.next()

            elif cursor.text == 'virtual':
                self.is_virtual = True
                cursor.next()

            elif cursor.type == TWord:
                if self.name:
                    cursor.error('Unexpected "%s"' % cursor.text)
                self.name = Name.parse(cursor)
                self.template_params = self.name.template_params

            else:
                if self.name:
                    return self
                cursor.error("Base type declaration has no name")


class AccessLabel(Node):

    def emit_interface(self, f):
        self.emit_line_directive(f)
        f.write(self.access + ':\n\n')

    @classmethod
    def _parse(cls, cursor):
        assert cursor.text in ['public', 'private', 'protected']

        self = AccessLabel()
        self.access = cursor.text

        cursor.next()
        if cursor.text != ':':
            cursor.error('Expected colon after "%s"' % cursor.text)

        cursor.next()
        return self


class TemplateParams(Node):

    dump_text = True

    @classmethod
    def _parse(cls, cursor):

        self = TemplateParams()

        assert cursor.text == '<'

        level = 0

        while cursor:
            if cursor.text == '<':
                level += 1
            elif cursor.text == '>':
                level -= 1
                if level == 0:
                    cursor.next()
                    break
                assert level >= 0
            cursor.next()

        return self


class Name(Node):

    @classmethod
    def _parse(cls, cursor):

        self = Name()
        self.scope = []
        self.identifier = None
        self.template_params = None

        another = True
        first = True

        while cursor:
            if another and cursor.type == TWord:
                self.scope.append(cursor.text)
                self.identifier = cursor.text
                another = False
                cursor.next()

            elif cursor.text == '::':
                if another and not first:
                    cursor.error("Repeated '::'")
                another = True
                cursor.next()

            elif self.identifier and (not another) and cursor.text == '<':
                if self.template_params is not None:
                    cursor.error("Extra template paramter list")
                self.template_params = TemplateParams.parse(cursor)

            else:
                if not self.identifier:
                    cursor.error("Expected a name")
                return self
            first = False


class RecordDefinition(Node):

    @classmethod
    def _parse(cls, cursor):

        self = RecordDefinition()
        self.record_kind = None
        self.name = None
        self.bases = None
        self.children = []
        self.attributes = []
        self.is_q_object = False

        self.head = TokenRange(cursor)

        while cursor:
            if cursor.text in ['class', 'struct', 'union', 'enum']:
                if self.record_kind is None:
                    self.record_kind = cursor.text
                elif self.record_kind == 'enum' and cursor.text == 'class':
                    self.record_kind = 'enum class'
                else:
                    cursor.error('Unexpected "%s" after "%s"' % (cursor.text, self.record_kind))
                cursor.next()

            elif cursor.type == TWord:
                if self.name is None:
                    name = Name.parse(cursor)
                    self.name = name.identifier
                else:
                    cursor.error('Unexpected "%s"; name is already "%s"' % (cursor.text, self.name))

            elif cursor.text == ':':
                if self.bases is not None:
                    cursor.error('Unexpected colon')

                cursor.next()
                base = BaseRecord.parse(cursor)
                if not base:
                    cursor.error('Expected base type after colon')

                self.bases = [base]

                while cursor.text == ',':
                    cursor.next()
                    base = BaseRecord.parse(cursor)
                    if not base:
                        cursor.error('Expected base type after comma')
                    self.bases.append(base)

                if cursor.text != '{':
                    cursor.error('Expected "{" after base type')
                break

            elif cursor.text == '{':
                break

            elif cursor.text == '[[':
                cursor.next()
                attr = cursor.text
                self.attributes.append(attr)
                cursor.next()
                if cursor.text != ']]':
                    cursor.error('Expected "]]" after "%s" to close attribute specifier' % attr)
                cursor.next()

            else:
                cursor.error(f'Unexpected "{cursor.text}"')


        # Parse body
        assert cursor.text == '{'

        head_end = cursor.copy()
        head_end.next()
        self.head.set_end(head_end)

        with append(local.RecordScope, self):
            if self.record_kind in ['enum', 'enum class']:
                self.enum_values = FunctionBody.parse(cursor)
            else:
                cursor.next()
                while cursor:
                    if cursor.text == '}':
                        cursor.next()
                        break

                    elif cursor.text in ['public', 'private', 'protected']:
                        label = AccessLabel.parse(cursor)
                        self.children.append(label)

                    elif qt_enabled and cursor.text == 'Q_OBJECT':
                        self.is_q_object = True
                        cursor.next()

                    else:
                        decl = Declaration.parse(cursor)
                        assert decl
                        self.children.append(decl)

                        # Decl could be function def (no ';') or normal def (';')
                        if cursor.text == ';':
                            cursor.next()

        return self

    def get_dependencies(self, names, typedef_names):
        deps = set()
        if self.bases:
            for base in self.bases:
                if len(base.name.scope) == 1 and base.name.identifier in names:
                    deps.add(base.name.identifier)

        for child in self.children:
            deps = deps | child.get_dependencies(names, typedef_names)

        return deps

    def emit_forward_declaration(self, f):
        if self.record_kind in ['enum', 'enum class']:
            self.emit_line_directive(f)

            for attr in self.attributes:
                f.write(f'[[f:{attr}]] ')
            f.write(self.text)
            f.write(';\n')
        else:
            f.write('%s %s;\n' % (self.record_kind, self.name))

    def emit_interface(self, f):


        if self.record_kind in ['enum', 'enum class'] and local.RecordScope:
            self.head.emit_line_directive(f)

            f.write(self.text)
            f.write('\n')

        elif self.record_kind in ['enum', 'enum class'] and not local.RecordScope:
            pass
        else:
            self.head.emit_line_directive(f)

            f.write(self.head.text.replace(self.record_kind, self.record_kind))
            f.write('\n')

            if self.is_q_object:
                f.write('Q_OBJECT\n\n')

            with append(local.RecordScope, self.name):
                for child in self.children:
                    child.emit_interface(f)

            f.write('}')

    def emit_inline_function_definitions(self, f):
        with append(local.RecordScope, self.name):
            for child in self.children:
                child.emit_inline_function_definitions(f)


class Specifier(Node):

    @classmethod
    def _parse(cls, cursor):

        start_cursor = cursor.copy()

        self = Specifier()
        self.record_kind = None
        self.record_definition = None
        self.record_start = None
        self.name = None
        self.template_params = None
        self.is_const = False
        self.is_static = False
        self.is_typedef = False
        self.is_virtual = False
        self.is_inline = False
        self.is_constexpr = False
        self.is_explicit = False
        self.is_extern_c = False
        self.attributes = []

        record_start = None

        while cursor:
            if cursor.text in ['class', 'struct', 'union', 'enum']:
                if self.record_kind is None:
                    self.record_kind = cursor.text
                    record_start = cursor.copy()
                elif self.record_kind == 'enum' and cursor.text == 'class':
                    self.record_kind = 'enum class'
                else:
                    cursor.error('Unexpected "%s" after "%s"' % (cursor.text, record_kind))
                cursor.next()

            elif cursor.text == 'const':
                if not self.is_const:
                    self.is_const = True
                    cursor.next()
                else:
                    cursor.error('Repeated "const"')

            elif cursor.text == 'explicit':
                if not self.is_explicit:
                    self.is_explicit = True
                    cursor.next()
                else:
                    cursor.error('Repeated "explicit"')

            elif cursor.text == 'static':
                if not self.is_static:
                    self.is_static = True
                    cursor.next()
                else:
                    cursor.error('Repeated "static"')

            elif cursor.text == 'typedef':
                if not self.is_typedef:
                    self.is_typedef = True
                    cursor.next()
                else:
                    cursor.error('Repeated "typedef"')

            elif cursor.text == 'virtual':
                if not self.is_virtual:
                    self.is_virtual = True
                    cursor.next()
                else:
                    cursor.error('Repeated "virtual"')

            elif cursor.text == 'inline':
                if not self.is_inline:
                    self.is_inline = True
                    cursor.next()
                else:
                    cursor.error('Repeated "inline"')

            elif cursor.text == 'constexpr':
                if not self.is_constexpr:
                    self.is_constexpr = True
                    cursor.next()
                else:
                    cursor.error('Repeated "constexpr"')

            elif cursor.text == '__stdcall':
                cursor.next()

            elif cursor.text == 'extern':
                cursor.next()
                if cursor.text != '"C"':
                    cursor.error('extern "C" is the only supported use of extern')
                cursor.next()
                self.is_extern_c = True

            elif cursor.text == 'template':
                cursor.next()
                if cursor.text != '<':
                    cursor.error('Expected "<" after "template"')

                self.template_params = TemplateParams.parse(cursor)

            elif self.record_kind and cursor.text in [':', '{']:
                cursor.set(record_start)
                self.record_definition = RecordDefinition.parse(cursor)
                break

            elif cursor.text == 'operator':
                # This must be a conversion operator; stop parsing the specifier
                break

            elif cursor.type == TWord or cursor.text == '::':
                # If this is a constructor, stop parsing the specifier
                if local.RecordScope and cursor.text == local.RecordScope[-1].name:
                    next_cursor = cursor.copy()
                    next_cursor.next()
                    if next_cursor.text == '(':
                        break

                if self.name is None:
                    self.name = Name.parse(cursor)
                else:
                    # We already have a name; this name must be the first declarator
                    break

            elif cursor.text == '~':
                # This is a destructor; stop parsing the specifier
                break

            elif cursor.text == '[[':
                cursor.next()
                attr = cursor.text
                self.attributes.append(attr)
                cursor.next()
                if cursor.text != ']]':
                    cursor.error('Expected "]]" after "%s" to close attribute specifier' % attr)
                cursor.next()


            else:
                if self.record_kind or self.name:
                    break
                cursor.error("Unexpected specifier component '%s'" % cursor.text)

        return self

    @property
    def text_without_attributes(self):
        return re.sub(r'\[\[.*?\]\]', '', self.text)


class FunctionBody(Node):
    @classmethod
    def _parse(cls, cursor):
        self = FunctionBody()

        assert cursor.text == '{'

        level = 0

        while cursor:
            if cursor.text == '{':
                level += 1
            elif cursor.text == '}':
                level -= 1
                if level == 0:
                    cursor.next()
                    return self
                if level < 0:
                    cursor.error("Unexpected '}'")
            cursor.next()

        cursor.error("Expected '}'")


class InitializerList(Node):
    @classmethod
    def _parse(cls, cursor):
        self = InitializerList()

        level = 0

        while cursor:
            if cursor.text in ['(', '[', '{', '<']:
                level += 1
            elif cursor.text in [')', ']', '}', '>']:
                level -= 1

            cursor.next()
            if level == 0 and cursor.text in ['{', ';']:
                return self
            if level < 0:
                cursor.error("Unexpected '}'")


class Declarator(Node):

    dump_text = True

    @classmethod
    def parse(cls, cursor):
        node = super().parse(cursor)
        if node.text == '()':
            cursor.error("'()' is not a valid declarator")
        return node
    
    @classmethod
    def _parse(cls, cursor):
        self = Declarator()

        level = 0
        while cursor:
            if level == 0 and cursor.text == '=':
                cursor.next()
                if cursor.text == '{':
                    level += 1
                    cursor.next()
                    continue

            if level == 0 and cursor.text in [',', ';', '{', ':']:
                break
            if cursor.text in ['(', '[', '{',]:
                level += 1
            elif cursor.text in [')', ']', '}',]:
                level -= 1
                if level < 0:
                    cursor.error("Unexpected '}'")
            cursor.next()

        return self

    @property
    def name(self):
        for token in self.range.tokens:
            if token.type == TWord:
                return token.text

    @property
    def reading(self):
        name_i = None
        tokens = self.range.tokens
        for i, token in enumerate(tokens):
            if token.type == TWord:
                name_i = i
                break

        if name_i is None:
            self.range.start_cursor.error("Cannot find declarator's name: " + self.text)

        l = r = name_i
        end = len(tokens)
        fwd = True

        while True:
            if fwd:
                r += 1
                if r >= end:
                    fwd = False
                elif tokens[r].text == ')':
                    fwd = False
                elif tokens[r].text == '=':
                    fwd = False
                elif tokens[r].text == '[':

                    while r < end and tokens[r].text != ']':
                        r += 1
                    yield 'array'

                elif tokens[r].text == '(':
                    level = 1
                    while r < end:
                        r += 1
                        if tokens[r].text == '(':
                            level += 1
                        if tokens[r].text == ')':
                            level -= 1
                            if level == 0:
                                break

                    yield 'function'
            else:
                l -= 1
                if l < 0:
                    break
                elif tokens[l].text == '*':
                    yield 'pointer'
                elif tokens[l].text == '&':
                    yield 'reference'
                elif tokens[l].text == '(':
                    fwd = True

        start_i = name_i

        
    @property
    def requires_full_declaration(self):
        for item in self.reading:
            if item in ['function', 'pointer', 'reference']:
                return False
        
        return True

    @property
    def extern_text(self):
        # The extern text for a declarator is the declarator without any initializer
        # and with any unnecessary brackets removed. For arrays with no explicit
        # length, we also count the elements in the initializer and add an explicit
        # length to the extern declaration so code seeing only the declaration can
        # make use of the array length information.
        text = ''

        in_init = False
        init_array_count = None
        
        # Remove balanced brackets, and count array initializer
        level = 0
        xlevel = 0
        for token in self.range.tokens:

            if token.text in '{[(':
                xlevel+=1

            if token.text in ')]}':
                xlevel-=1

            if (token.text == '(' or (level > 0 and token.text in '{[')): level += 1
            elif level != 0 and token.text in ']})': level -= 1
            elif level == 0:
                text += token.text + ' '

            if level == 0 and init_array_count is None and token.text == '=':
                in_init = True

            if xlevel == 0 and init_array_count is None and in_init and token.text == '{':
                init_array_count = 0

            if xlevel == 1 and in_init and token.text == ',':
                if init_array_count is None:
                    init_array_count = 1
                init_array_count += 1

        # Remove =-assignment and add array count
        if 'operator' not in text:
            if '=' in text:
                text, initializer = text.split('=', maxsplit=1)
                text = text.rstrip()
                initializer = initializer.lstrip()
                if text.endswith('[ ]') and init_array_count is not None:
                    text = text[:-3] + f'[{init_array_count}]'

        return text


class NamespaceDeclaration(Node):

    @classmethod
    def _parse(cls, cursor):

        self = NamespaceDeclaration()
        self.children = []
        self.manual_deps = []

        assert cursor.text == 'namespace'

        cursor.next()

        if cursor.type != TWord:
            cursor.error("Expected namespace name")

        self.name = cursor.text

        cursor.next()

        if cursor.text != "{":
            cursor.error("Expected '{' after namespace name")

        cursor.next()

        while cursor and cursor.text != "}":
            child = parse_declaration(cursor)
            self.children.append(child)

        if cursor.text != "}":
            cursor.error("Expected '}' to close namespace")
        cursor.next()

        return self

    def emit_interface(self, f):
        self.emit_line_directive(f)
        f.write('namespace %s {\n\n' % self.name)

        for child in self.children:
            child.emit_forward_declaration(f)
            child.emit_interface(f)

        f.write('}\n\n')

    def emit_implementation(self, f):
        self.emit_line_directive(f)
        f.write('namespace %s {\n\n' % self.name)

        for child in self.children:
            child.emit_implementation(f)

        f.write('}\n\n')

    def emit_inline_function_definitions(self, f):

        start = f.tell()

        self.emit_line_directive(f)
        f.write('namespace %s {\n\n' % self.name)

        check = f.tell()

        for child in self.children:
            child.emit_inline_function_definitions(f)

        # Avoid empty ns if nothing got written
        if f.tell() == check:
            f.seek(start, os.SEEK_SET)
            f.truncate()
            return
        
        f.write('}\n\n')

    @property
    def defined_names(self):
        return [self.name]

    def get_dependencies(self, names, typedef_names):
        deps = set()
        for child in self.children:
            deps = deps | child.get_dependencies(names, typedef_names)
        deps |= set(self.manual_deps)
        return deps


class Declaration(Node):

    @classmethod
    def _parse(cls, cursor):
        self = Declaration()
        self.specifier = None
        self.initializer_list = None
        self.function_body = None
        self.declarators = []
        self.manual_deps = []

        self.specifier = Specifier.parse(cursor)

        if cursor.text == ';':
            cursor.next()
            return self

        try:
            declarator = Declarator.parse(cursor)

        # Handy diagnostic for misnamed ctors
        except ParseError as e:
            if '()' in str(e) and local.RecordScope:
                expected_name = local.RecordScope[-1].name
                got_name = self.specifier.name.text
                cursor.error('Constructor must be called %s, not %s' % (expected_name, got_name))
            else:
                raise

        self.declarators.append(declarator)

        if cursor.text == ':':
            self.initializer_list = InitializerList.parse(cursor)

        if cursor.text == '{':
            self.function_body = FunctionBody.parse(cursor)
            return self

        while cursor.text == ',':
            cursor.next()
            declarator = Declarator.parse(cursor)
            self.declarators.append(declarator)

        if cursor.text != ';':
            cursor.error("Expected ';'")

        cursor.next()

        return self

    @property
    def is_inline_or_template_function(self):
        return self.function_body and (
            self.specifier.is_inline or (
                self.specifier.template_params and
                not local.RecordScope
            )
        )

    def emit_forward_declaration(self, f):
        if not self.specifier.template_params:
            record = self.specifier.record_definition
            if record:
                record.emit_forward_declaration(f)

    def emit_inline_function_definitions(self, f):
        if self.is_inline_or_template_function:
            self.emit_function_definition(f)
            return
        
        record = self.specifier.record_definition
        if record:
            record.emit_inline_function_definitions(f)

    def emit_interface(self, f):

        if self.specifier.is_constexpr:

            # As a convenience, we allow omission of "static" on a member
            # constexpr var, because they always must be static.
            if local.RecordScope and not self.specifier.is_static:
                f.write('static ')

            f.write(self.text + '\n\n')
            return

        if self.specifier.is_typedef:
            f.write(self.text)
            f.write('\n')

        # If this is a function:
        if self.function_body:
            self.emit_function_declaration(f)
            return

        # If this is static member vars:
        if self.specifier.is_static:
            self.emit_static_member_vars_interface(f)
            return

        if self.specifier.record_definition and self.specifier.template_params:
            self.emit_line_directive(f)
            f.write(self.text)
            f.write('\n\n')
            return

        if self.specifier.is_typedef:
            return

        # If this is a record definition: 
        record = self.specifier.record_definition
        if record:
            pos = f.tell()

            record.emit_interface(f)
            f.write(', '.join(d.text for d in self.declarators))
            if f.tell() != pos:
                f.write(';\n\n')
            return

        # If this is a record member: 
        if local.RecordScope:
            self.emit_line_directive(f)
            spec = self.specifier.text
            if spec:
                f.write(spec + ' ')

            f.write(', '.join(d.text for d in self.declarators))
            f.write(';\n\n')
            return
        
        # Otherwise, it's types & vars:
        if not self.declarators:
            raise Exception(self.line_directive)

        self.emit_line_directive(f)

        # Don't double "extern" if extern "C"
        if self.specifier.is_extern_c:
            f.write(self.specifier.text)
        else:
            f.write('extern %s' % self.specifier.text_without_attributes)

        for i, decl in enumerate(self.declarators):
            if i != 0:
                f.write(', ')
            f.write(' %s' % decl.extern_text)
        f.write(';\n')

    def emit_function_declaration(self, f):
        self.emit_line_directive(f)
        spec = self.specifier.text
        if spec:
            f.write(spec + ' ')
        f.write(self.declarators[0].text)
        f.write(';\n\n')

    def emit_static_member_vars_interface(self, f):
        self.emit_line_directive(f)

        if self.specifier.is_constexpr:
            f.write(self.text + '\n\n')
            return
            
        spec = self.specifier.text
        if spec:
            f.write(spec + ' ')

        declarator_texts = []
        for i, d in enumerate(self.declarators):
            if i != 0:
                f.write(', ')
            for token in d.range.tokens:
                if token.text == '=':
                    break
                f.write(token.text_with_whitespace)

        f.write(';\n\n')

    def emit_static_member_vars_implementation(self, f):

        self.emit_line_directive(f)

        for token in self.specifier.range.tokens:
            if token.text != 'static':
                f.write(token.text_with_whitespace)

        for i, declarator in enumerate(self.declarators):

            if i != 0:
                f.write(', ')
                
            scoped = False
            for token in declarator.range.tokens:
                if self.specifier.is_constexpr and token.text == '=':
                    break
                if (not scoped) and token.text == declarator.name:
                    f.write('::'.join(local.RecordScope + ['']))
                    scoped = True
                f.write(token.text_with_whitespace)

        f.write(';\n\n')

    def emit_function_definition(self, f):
        self.emit_line_directive(f)

        for token in self.specifier.range.tokens:
            if token.text not in ['virtual', 'static', 'explicit']:
                f.write(token.text_with_whitespace)

        declarator, = self.declarators
        scoped = False
        skip_default_arg = False
        in_arg_list = False
        for token in declarator.range.tokens:

            if token.text == '(':
                in_arg_list = True

            if token.text == '=' and in_arg_list:
                skip_default_arg = True

            if skip_default_arg:
                if token.text in [',', ')']:
                    skip_default_arg = False
                else:
                    continue
                
            if token.text != 'override':
                if (not scoped) and (token.text == declarator.name or token.text == '~'):
                    f.write('::'.join(local.RecordScope + ['']))
                    scoped = True
                f.write(token.text_with_whitespace)

        if self.initializer_list:
            f.write(self.initializer_list.text)

        f.write(unindent(self.function_body.text))

        f.write('\n\n')
        
    def emit_implementation(self, f):

        if self.specifier.is_constexpr and not local.RecordScope:
            return
        
        if self.function_body:
            if not self.is_inline_or_template_function:
                self.emit_function_definition(f)
            return

        if self.specifier.is_static:
            self.emit_static_member_vars_implementation(f)
            return

        if self.specifier.is_constexpr and local.RecordScope:
            self.emit_static_member_vars_implementation(f)
            return

        if self.specifier.record_definition and self.specifier.template_params:
            return

        if self.specifier.is_typedef:
            return

        record = self.specifier.record_definition
        if record:
            with append(local.RecordScope, record.name):
                for child in record.children:
                    child.emit_implementation(f)
            return

        if not local.RecordScope:
            self.emit_line_directive(f)
            f.write(self.text)
            f.write('\n')

    @property
    def defined_names(self):
        record = self.specifier.record_definition
        if record:
            yield record.name

        for declarator in self.declarators:
            yield declarator.name

    def get_dependencies(self, names, typedef_names):
        r = set()

        record = self.specifier.record_definition
        if record:
            r = r | record.get_dependencies(names, typedef_names)

        # Does the specifier refer to a name?
        spec_deps = set()

        ident = self.specifier and self.specifier.name and self.specifier.name.identifier
        if ident and ident in names:
            spec_deps.add(ident)

        template_params = (
            self.specifier and
            self.specifier.name and
            self.specifier.name.template_params
        )
        if template_params:
            for token in template_params.range.tokens:
                if token.text in names:
                    spec_deps.add(token.text)

        # If so, and any declarators aren't pointers etc., there's a dependency
        if spec_deps:
            if any(
                d.requires_full_declaration
                for d in self.declarators
            ):
                r = r | spec_deps


        # If any of them are typedefs, it's a dependency
        r = r | (spec_deps & typedef_names)


        # Normally, function arguments do not create interface dependencies to their
        # types (a forward declaration is enough). But, for typedefs, we depend on the typedef.
        if typedef_names:

            declarator_words = set(
                token.text
                for declarator in self.declarators
                for token in declarator.range.tokens
                if token.type == TWord
            )

            used_typedefs = typedef_names & declarator_words
            if used_typedefs:
                r |= used_typedefs
            

        r |= set(self.manual_deps)

        return r


class ObjCProperty(Node):

    @classmethod
    def _parse(cls, cursor):

        self = ObjCProperty()

        assert cursor.text == '@property'

        while cursor and cursor.text != ';':
            cursor.next()

        assert cursor.text == ';'
        cursor.next()
        return self


class ObjCMethod(Node):

    @classmethod
    def _parse(cls, cursor):

        self = ObjCMethod()

        level = 0

        while cursor:
            if cursor.text == '{':
                level += 1
            elif cursor.text == '}':
                level -= 1
                if level == 0:
                    cursor.next()
                    break
                elif level < 0:
                    cursor.error("Unexpected '}'")
            cursor.next()

        return self

    def emit_interface(self, f):
        tokens = self.range.tokens
        self.emit_line_directive(f)
        for i, token in enumerate(tokens):
            j = i + 1
            if j < len(tokens) and tokens[j].text == '{':
                f.write(token.text)
                f.write(';\n\n')
                return

            f.write(token.text_with_whitespace)

    def emit_implementation(self, f):
        self.emit_line_directive(f)
        f.write(self.text)
        f.write('\n\n')


class ObjCProtocol(Node):

    @classmethod
    def _parse(cls, cursor):

        self = ObjCProtocol()

        self.name = None

        # Parse head
        assert cursor.text == '@protocol'
        cursor.next()

        self.name = cursor.text
        cursor.next()

        while cursor.text != '@end':
            cursor.next()
        cursor.next()

        self.defined_names = [self.name]

        return self

    def emit_interface(self, f):
        f.write('\n')
        f.write(self.line_directive)
        f.write(self.text)
        f.write('\n')


class ObjCClass(Node):

    @classmethod
    def _parse(cls, cursor):

        self = ObjCClass()

        self.name = None
        self.methods = []
        self.properties = []
        self.base = None
        self.children = []

        # Parse head
        assert cursor.text == '@interface'

        self.head = TokenRange(cursor)

        cursor.next()
        if cursor.type != TWord:
            cursor.error("Expected class name")

        self.name = cursor.text
        cursor.next()

        level = 0

        while cursor:
            if level == 0 and cursor.text == ':':
                cursor.next()
                self.base = cursor.text
            elif cursor.text == '{':
                level += 1
            elif cursor.text == '}':
                level -= 1
                if level == 0:
                    cursor.next()
                    break
                elif level < 0:
                    cursor.error("Unexpected '}'")
            elif cursor.text == '@end':
                break
            elif level == 1:
                decl = Declaration.parse(cursor)
                assert decl
                self.children.append(decl)
                continue
            cursor.next()

        self.head.set_end(cursor)

        # Skip any method declarations; we'll declare all methods
        while cursor and cursor.text != '@end':

            if cursor.text == '@property':
                prop = ObjCProperty.parse(cursor)
                self.properties.append(prop)
            else:
                cursor.next()

        if cursor.text != '@end':
            cursor.error("Expected '@end'")

        cursor.next()

        # Parse body
        if cursor.text != '@implementation':
            cursor.error("Expected '@implementation'")

        cursor.next()

        if cursor.text != self.name:
            cursor.error("Expected this @implementation to be for %s" % self.name)

        cursor.next()

        while cursor:
            if cursor.text in ['-', '+']:
                self.methods.append(ObjCMethod.parse(cursor))
            elif cursor.text == '@end':
                cursor.next()
                break
            else:
                cursor.error("Expected ObjC method or @end, not '%s'" % cursor.text)

        return self

    def emit_forward_declaration(self, f):
        f.write('@class %s;\n' % self.name)
        
    def emit_interface(self, f):
        self.head.emit_line_directive(f)
        f.write(self.head.text)
        f.write('\n\n')
        for prop in self.properties:
            prop.emit_line_directive(f)
            f.write(prop.text)
            f.write('\n\n')
        for method in self.methods:
            method.emit_interface(f)
        f.write('@end\n\n')

    def emit_implementation(self, f):
        self.emit_line_directive(f)
        f.write('@implementation %s\n\n' % self.name)
        for method in self.methods:
            method.emit_implementation(f)
        f.write('@end\n')

    @property
    def defined_names(self):
        return [self.name]

    def get_dependencies(self, names, typedef_names):
        deps = set()
        if self.base:
            if self.base in names:
                deps.add(self.base)

        for child in self.children:
            deps = deps | child.get_dependencies(names, typedef_names)

        return deps


def parse_declaration(cursor):
    if cursor.text == '@interface':
        return ObjCClass.parse(cursor)
    elif cursor.text == '@protocol':
        return ObjCProtocol.parse(cursor)
    elif cursor.text == 'namespace':
        return NamespaceDeclaration.parse(cursor)
    else:
        return Declaration.parse(cursor)


class Database:
    def __init__(self, debug_lexer, debug_syntax):
        self.debug_lexer = debug_lexer
        self.debug_syntax = debug_syntax
        self.includes = []
        self.decls = []

    def emit_interfaces(self, f):

        for include in self.includes:
            with open(include, 'rt') as i:
                f.write(i.read())
                f.write('\n')
            
        for decl in self.decls:
            decl.emit_forward_declaration(f)
        f.write('\n')

        for decl in self.decls:
            decl.emit_interface(f)
        f.write('\n')

        for decl in self.decls:
            decl.emit_inline_function_definitions(f)
        f.write('\n')

    def emit_implementations(self, f):
        for decl in self.decls:
            decl.emit_implementation(f)

    def parse_input(self, path):
        local.RecordScope = []

        tokens = Lexer(path)

        manual_deps = []
        decls = []

        if self.debug_lexer:
            cursor = Cursor(tokens, 0)
            while cursor:
                token = cursor.token
                print(str(token.line).ljust(5), token.type.ljust(12), token.text, file=sys.stderr)
                cursor.next()


        cursor = Cursor(tokens, 0)

        while cursor and cursor.text == '#pragma':
            cursor.next()
            if cursor.text != 'depends':
                cursor.error("Unrecognized #pragma: '%s'" % cursor.text)
            cursor.next()
            manual_deps.append(cursor.text)
            cursor.next()

        while cursor:
            decl = parse_declaration(cursor)
            decl.manual_deps = manual_deps
            decls.append(decl)

            if self.debug_syntax:
                decl.dump()

        return decls

    def parse(self, input_paths, manifest_path):

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            decl_lists = list(pool.map(self.parse_input, input_paths))

        for decl_list in decl_lists:
            self.decls.extend(decl_list)

        if manifest_path:
            with open(manifest_path, 'wt') as f:
                for path, decl_list in zip(input_paths, decl_lists):
                    f.write(path + ':')
                    for decl in decl_list:
                        f.write(' ' + ' '.join(decl.defined_names))
                    f.write('\n')
                
        

    def emit_all(self, source_path, header_path, depinfo_path, includes):
        self.includes = includes
        self.depinfo_path = depinfo_path

        # Figure out interface dependencies
        self.sort_decls()

        # Parse all declarations
        # Emit output
        if header_path:
            with to_file(header_path, 'wt') as f:
                f.write('#pragma once\n\n')
                self.emit_interfaces(f)

        if source_path:
            with to_file(source_path, 'wt') as f:
                if not header_path:
                    self.emit_interfaces(f)
                    f.write('\n// -- IMPLEMENTATIONS --\n\n')
                self.emit_implementations(f)

    def sort_decls(self):
        decls = self.decls

        names = set()
        typedef_names = set()
        decls_by_name = defaultdict(set)
        for decl in decls:
            for name in decl.defined_names:
                names.add(name)
                decls_by_name[name].add(decl)

                if isinstance(decl, Declaration) and decl.specifier.is_typedef:
                    typedef_names.add(name)

        decl_deps = set()
        for decl in decls:
            for name in decl.defined_names:
                for dep in decl.get_dependencies(names, typedef_names):
                    for dep_decl in decls_by_name[dep]:
                        if dep_decl is not decl:
                            decl_deps.add((dep_decl, decl))

        deps_by_decl = defaultdict(set)
        for a, b in decl_deps:
            deps_by_decl[b].add(a)

        if self.depinfo_path:
            with open(self.depinfo_path, 'wt') as f:
                for b, a in deps_by_decl.items():
                    for bn in b.defined_names:
                        ans = []
                        for ad in a:
                            ans.extend(ad.defined_names)

                        f.write(bn + ': ' + ' '.join(ans) + '\n')

        done = set()
        sorted_decls = []

        while decls:
            moving = False
            for decl in decls:
                ok = True
                for dep in deps_by_decl[decl]:
                    if dep not in done:
                        ok = False
                        break
                if ok:
                    done.add(decl)
                    sorted_decls.append(decl)
                    decls.remove(decl)
                    moving = True
                    break

            if not moving:
                msg = 'Cycle in dependencies:\n'
                for decl in decls:
                    for dep in deps_by_decl[decl]:
                        msg += '%s -> %s\n' % (
                            ', '.join(dep.defined_names),
                            ', '.join(decl.defined_names)
                        )
                raise Exception(msg)

        self.decls = sorted_decls

        #print('---')
        #for decl in sorted_decls:
        #    print(list(decl.defined_names))
        #    if list(decl.defined_names) == []:
        #        print('--', decl.text)


unindent_re = re.compile(r'\n +')


def unindent(text):
    ms = unindent_re.findall(text)
    if ms:
        shortest = min(ms, key=len)
        return text.replace(shortest, '\n')
    return text


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-oc', metavar='PATH', help='Output source file')
    parser.add_argument('-oh', metavar='PATH', help='Output header file')
    parser.add_argument('-od', metavar='PATH', help='Output dependency information')
    parser.add_argument('-on', metavar='PATH', help='Output names declared in each file')
    parser.add_argument('-i', metavar='PATH', help='Includes to add to generated source', nargs='*')
    parser.add_argument('-nl', action='store_true', help='Omit #line directives')
    parser.add_argument('-qt', action='store_true', help='Enable Qt extensions')
    parser.add_argument('inputs', nargs='+', help='Input files')
    
    debug = parser.add_argument_group('arguments used for debugging')
    debug.add_argument('-dt', action='store_true', help='Trace tokenization to stderr')
    debug.add_argument('-ds', action='store_true', help='Trace syntax tree to stderr')
    debug.add_argument('-dx', action='store_true', help='Print tracebacks on parse errors')

    return parser


def parse_to_db(args):
    if args.nl:
        global line_directives_enabled
        line_directives_enabled = False

    if args.qt:
        global qt_enabled
        qt_enabled = True

    try:
        db = Database(args.dt, args.ds)
        db.parse(args.inputs, args.on)
    except ParseError as e:
        print(e.args[0], file=sys.stderr)
        for item in e.record_scope:
            print('While parsing ', item, file=sys.stderr)
        if args.dx:
            raise
        else:
            sys.exit(1)

    return db


def main():
    parser = get_argument_parser()
    args = parser.parse_args()

    if args.oc is None and args.oh is None:
        parser.error('at least one of -oc and -oh must be specified.')

    db = parse_to_db(args)
    db.emit_all(args.oc, args.oh, args.od, args.i or [])
    return db

if __name__ == '__main__':
    main()
