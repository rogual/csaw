
csaw: life's too short to write header files
--------------------------------------------


What It Is
----------

A preprocessor for C++. You could also consider it to be
a compiler for an (unnamed) dialect of C++.


Why?
----

I want to be able to write code as a set of files, each file
containing some arbitrary declarations, not write any #include
or #imports, not write anything twice (like .c/.h files) and
have the computer sort it all out.


Basic Usage
-----------

csaw -oc <output_src> -oh <output_h> <input_files>

You give csaw a set of input files. Each input file contains
one or more C++ declarations. Think of each file as a module.
You don't need to #include or otherwise import anything; if
a module references an identifier defined in another module,
the dependency is resolved automatically.

Csaw outputs a single C++ source file containing all
your code. Interfaces and implementations are split apart
automatically and written to the file in dependency order.

Optionally, you can ask for a separate .h file containing
only the interfaces, and omitting most function bodies. This
is useful if you're compiling a library and you want to
distribute a normal .h file to your users.


Features
--------

Supports C, C++ and Objective-C.


Libraries
---------

Csaw is not designed to process arbitrary C or C++ code and you
shouldn't feed libraries to it. Instead, process only your own
code with csaw, and #include your libraries using csaw's -i option
or your compiler's command-line switches.


Preprocessor
------------

Csaw doesn't implement a preprocessor or understand preprocessor
directives. If you use the preprocessor, you should run it first
and pass the processed source to csaw.


Limitations: General
--------------------

* The whole thing is a massive hack.


Limitation: Nested Names
------------------------

Nested classes are not fully supported. The difficulty comes from the
fact that nested classes cannot be forward-declared outside of their
containing class.

This causes two problems; one fixable, one not.

1. We rely on blanket forward-declaring all classes at the
top of the output file, so we know the names will be there
for the rest of the file. This means we don't have to think
about "soft dependencies" (requiring forward-declaration),
only "interface dependencies" (requiring full declaration).
But, since there is no forward-declaration of nested
classes, we must consider ANY reference at all to the name
of a nested class to introduce an interface dependency. This
is MUCH more work, since names can appear basically anywhere.

2. The situation is not even solvable in the general case.

For example, it is not possible in C++ to order these
definitions in a way that will satisfy a conformant compiler:

class A {
  class AA {};
  void f(B::BB) {}
};

class B {
  class BB {};
  void f(A::AA) {}
};

Avoid using nested names from outside their containing class.


Limitations: Namespaces
-----------------------

Anonymous namespaces are not supported.


Limitation: Static const member variables
-----------------------------------------

C++ allows static const integral member variables to be used
as constants; csaw supports this but requires you to define
them 'constexpr'.

Example:

class MyClass {
  static const int sz = 16;
  //     ^ csaw requires 'constexpr' here
  int array[sz];
}

Rationale: This so csaw doesn't have to figure out what is
an integer and what isn't (C++ rules only make this provision
for integral types)


Enhancement: Classes
--------------------

Class member functions should always be defined in the class
body. In C++ this also makes them inline, but with csaw this
will only happen if you explicity mark them 'inline'.

Rationale: Saves you writing the function signature twice.


Enhancement: Static member variables
------------------------------------

C++ requires (most) static member variables to be initialized
out-of-line. With csaw, you can always initialize them inline.

Example:

class MyClass {
  static const char *word = "hippo"; // OK; would be invalid C++
};

Rationale: Saves you declaring the variable twice.


Enhancement: Objective-C
------------------------

Objective-C classes may be defined in a single @interface block and
do not require a separate @implementation block:

@interface MyClass {
  int _myVar;
}

-(void)myMethod {
  puts("hi");
}
@end

Rationale: Saves you a bit of typing.


Limitation: Objective-C
-----------------------

If you choose to write your Objective-C classes in the normal
@interface + @implementation style, then each Objective-C class must
be written as an @interface block with its @implementation immediately
following it.


Limitation: Templates
---------------------

Templates are supported, but some conservative estimates are
made about what depends on what. For example, with:

class my_class {
    my_template<my_type> my_var;
};

my_class will be considered to depend on my_type being fully
defined, though that won't necessarily be the case; my_template
could instantiate only a pointer to it, or not use it at all.

(my_class will also be considered to depend on my_template,
but that is always correct.)

It doesn't seem feasible to automatically do the right thing
in this case, though we could probably have csaw take hints
on what to do for a particular template.


Q&A
---

Q: Why are you parsing C++ by hand? Are you insane? Do you know
libclang exists?

A: Clang only reliably parses correct C++, and the whole point
of csaw is to save you from having to write correct C++. In
particular, clang requires type information to be declared
before names are referenced, which we don't want to require.

See Also
--------

Lazy C++ <https://www.lazycplusplus.com> : Another project
with a similar goal.

