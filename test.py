#!/usr/bin/env python3

import csaw
import sys
import io
import re

from collections import namedtuple

spec = open('test.org', 'rt')


space_re_1 = re.compile(r'(?<=[A-Za-z0-9_])\s+(?=[A-Za-z0-9_])')
space_re_0 = re.compile(r'\s+')
strip_re = re.compile('^#pragma once$|#line.*$', re.MULTILINE)
header_re = re.compile(r'\*+ ')


def normalize(s):
    marker = '\0'
    s = strip_re.sub('', s)
    s = space_re_1.sub(marker, s)
    s = space_re_0.sub('', s)
    s = s.replace(marker, ' ')
    s = s.strip()

    # Stop the "C" in extern "C" being squashed up against
    # the next token and making a suffixed string
    s = s.replace('"C"', '"C" ')

    return s


def blocks():
    level = 0
    block = ''
    title = ''
    for line in spec:
        if header_re.match(line):
            yield level, title, block
            title = line.lstrip('*')
            level = len(line) - len(title)
            title = title.strip()
            block = ''
        else:
            block += line
    yield level, title, block


def items(): 
    cat = None
    test = None
    in_ = None
    dep = None
    header = ''
    source = ''
    last_level = None
    for level, title, block in blocks():
        if level == 1 or level == 2 and last_level >= level:
            if test:
                yield cat, test, dep, in_, header, source
                test = None
                in_ = None
                dep = None
                header = ''
                source = ''

        if level == 1:
            cat = title
        elif level == 2:
            test = title
        elif level == 3:
            if title == 'dep':
                dep = block
            elif title == 'in':
                in_ = block
            elif title == 'header':
                header = block
            elif title == 'source':
                source = block
        last_level = level
    if test:
        yield cat, test, dep, in_, header, source


def check_compiles(code):
    import subprocess

    with open('/tmp/csaw.test.cc', 'wt') as f:
        f.write(code)

    subprocess.check_call(
        ['/usr/bin/c++',
         '/tmp/csaw.test.cc',
         '-fsyntax-only',
         '-fdeclspec',
         '-std=c++14',
         '-Wno-unknown-attributes',
         '-Wno-pragma-once-outside-header',
         '-Wno-ignored-attributes',
        ]
    )


query = ''
words = [x for x in sys.argv[1:] if not x.startswith('-')]
if words:
    query = words[0]
    

for cat, test, dep, in_, header, source in items():

    if query not in cat and query not in test:
        continue

    inputs = [
        io.StringIO(in_)
    ]

    debug_lexer = '-dl' in sys.argv
    debug_syntax = '-ds' in sys.argv
    
    db = csaw.Database(debug_lexer, debug_syntax)
    db.parse(inputs, None)

    raw_output_source_file = io.StringIO()
    raw_output_header_file = io.StringIO()

    db.emit_all(raw_output_source_file, raw_output_header_file, None, [])

    raw_output_header = raw_output_header_file.getvalue()
    raw_output_source = raw_output_source_file.getvalue()

    output_source = normalize(raw_output_source)
    output_header = normalize(raw_output_header)

    source = normalize(source)
    header = normalize(header)

    ok = True
    title = cat + ': ' + test

    expect_fail = test.startswith('TODO ')

    matches = output_source == source and output_header == header

    if expect_fail:
        if matches:
            print('[FAIL]', title)
            print('Test passed unexpectedly')
            ok = False
        else:
            print('[TODO]', title)
            ok = True

    else:
        if matches:
            print('[ OK ]', title)
        else:
            print('[FAIL]', title)
            if output_source != source:
                print('SOURCE differs:')
                print('Expected:', source)
                print('----')
                print('Got     :', output_source)
                ok = False

            if output_header != header:
                print('HEADER differs:')
                print('Expected:', header)
                print('----')
                print('Got     :', output_header)
                ok = False

        if 'objc' not in cat:
            try:
                check_compiles((dep or '') + '\n' + raw_output_header + '\n' + raw_output_source)
            except Exception as e:
                print(e)
    
