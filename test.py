#!/usr/bin/env python3

import csaw
import io
import re

from collections import namedtuple

spec = open('test.org', 'rt')


space_re_1 = re.compile(r'(?<=[A-Za-z0-9_])\s+(?=[A-Za-z0-9_])')
space_re_0 = re.compile(r'\s+')
strip_re = re.compile('^#pragma once$|#line.*$', re.MULTILINE)


def normalize(s):
    marker = '\0'
    s = strip_re.sub('', s)
    s = space_re_1.sub(marker, s)
    s = space_re_0.sub('', s)
    s = s.replace(marker, ' ')
    s = s.strip()
    return s


def blocks():
    level = 0
    block = ''
    title = ''
    for line in spec:
        if line.startswith('*'):
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
    header = ''
    source = ''
    last_level = None
    for level, title, block in blocks():
        if level == 1 or level == 2 and last_level >= level:
            if test:
                yield cat, test, in_, header, source
                test = None
                in_ = None
                header = ''
                source = ''

        if level == 1:
            cat = title
        elif level == 2:
            test = title
        elif level == 3:
            if title == 'in':
                in_ = block
            elif title == 'header':
                header = block
            elif title == 'source':
                source = block
        last_level = level
    if test:
        yield cat, test, in_, header, source

for cat, test, in_, header, source in items():
    #print (cat, test, in_, header, source)

    inputs = [
        io.StringIO(in_)
    ]

    db = csaw.Database(False, False)
    db.parse(inputs, None)

    output_source = io.StringIO()
    output_header = io.StringIO()

    db.emit_all(output_source, output_header, None, [])

    output_header = output_header.getvalue()
    output_source = output_source.getvalue()

    output_source = normalize(output_source)
    output_header = normalize(output_header)

    source = normalize(source)
    header = normalize(header)

    ok = True
    print(cat, test, end=' ')

    if output_source != source:
        print('SOURCE differs:')
        print(source)
        print('----')
        print(output_source)
        ok = False

    if output_header != header:
        print('HEADER differs:')
        print(header)
        print('----')
        print(output_header)
        ok = False

    if ok:
        print('OK')

    
