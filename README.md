# uslang

An unorthodox scripting language written in C++.

```uslang
@proj = "Unorthodox Scripting Language"
@desc = "hybrid-typed, object-oriented"

println "${proj} is a ${desc} scripting language."
```

## Build Instructions

To build, open a terminal, navigate to the source code, and run:

```bash
make all
```

## History

I began writing this scripting language as a junior in high school.

It began as a side project to build an alternative shell for Linux and evolved into what it is today.

## TODO

Stay tuned for updates and additions to this README.

## Testing

Below is the test script I am using to test the language features with each commit.

To execute it, use:
```shell
make test
```

## Test Code
```
##################################################

println "\n+------------+"
println "| Arithmetic |"
println "+------------+"

##################################################

# addition
@a = 10
@b = 20
@c = @a + @b
println "${@a} + ${@b} = ${@c}"

# subtraction
@a = 10
@b = 20
@c = @a - @b
println "${@a} - ${@b} = ${@c}"

# multiplication
@a = 10
@b = 20
@c = @a * @b
println "${@a} * ${@b} = ${@c}"

# division
@a = 10.0
@b = 20
@c = @a / @b
println "${@a} / ${@b} = ${@c}"

##################################################

println "\n+-----------------------+"
println "| Arithmetic Expression |"
println "+-----------------------+"

##################################################

@a = 10.0
@b = 20.0
@c = (((@a * 2) / @b) + 9)
println "((${@a} * 2) / ${@b}) + 9 = ${@c}"

##################################################

println "\n+----------------------+"
println "| String Concatenation |"
println "+----------------------+"

##################################################

# simple concatenation
@a = "Hello"
@b = " World"
@c = @a + @b
println "'${@a}' + '${@b}' = '${@c}'"

# concatenation expression
@a = "Hello"
@b = "World"
@c = (@a + " " + @b)
println "'${@a}' + ' ' + '${@b}' = '${@c}'"

# string multiplication
@a = "Hello"
@b = 5
@c = @a * @b
println "'${@a}' * ${@b} = '${@c}'"

##################################################

println "\n+------------+"
println "| Logical OR |"
println "+------------+"

##################################################

@a = false
@b = false
@c = @a || @b
println "${@a} || ${@b} = ${@c}"

@a = false
@b = true
@c = @a || @b
println "${@a} || ${@b} = ${@c}"

@a = true
@b = false
@c = @a || @b
println "${@a} || ${@b} = ${@c}"

@a = true
@b = true
@c = @a || @b
println "${@a} || ${@b} = ${@c}"

##################################################

println "\n+-------------+"
println "| Logical AND |"
println "+-------------+"

##################################################

@a = false
@b = false
@c = @a && @b
println "${@a} && ${@b} = ${@c}"

@a = false
@b = true
@c = @a && @b
println "${@a} && ${@b} = ${@c}"

@a = true
@b = false
@c = @a && @b
println "${@a} && ${@b} = ${@c}"

@a = true
@b = true
@c = @a && @b
println "${@a} && ${@b} = ${@c}"

##################################################

println "\n"
println "+-------------+"
println "| Logical NOT |"
println "+-------------+"

##################################################

@a = true
@b = !@a
println "!${@a} = ${@b}"

@a = !@b
println "!${@b} = ${@a}"

@a = true
@b = true
@c = !(@a == @b)
@d = !(@a != @b)
println "!(${@a} == ${@b}) = ${@c}"
println "!(${@a} != ${@b}) = ${@d}"

##################################################

println "\n"
println "+------------------+"
println "| Relational Logic |"
println "+------------------+"

##################################################

@a = 10
@b = 20
@c = @a < @b
println "${@a} < ${@b} = ${@c}"
@a = 10
@b = 10
@c = @a < @b
println "${@a} < ${@b} = ${@c}"
@a = 20
@b = 10
@c = @a < @b
println "${@a} < ${@b} = ${@c}"
@a = 10
@b = 10
@c = @a <= @b
println "${@a} <= ${@b} = ${@c}"
@a = 20
@b = 10
@c = @a <= @b
println "${@a} <= ${@b} = ${@c}"

@a = 10
@b = 20
@c = @a > @b
println "${@a} > ${@b} = ${@c}"
@a = 10
@b = 10
@c = @a > @b
println "${@a} > ${@b} = ${@c}"
@a = 20
@b = 10
@c = @a > @b
println "${@a} > ${@b} = ${@c}"
@a = 10
@b = 10
@c = @a >= @b
println "${@a} >= ${@b} = ${@c}"
@a = 20
@b = 10
@c = @a >= @b
println "${@a} >= ${@b} = ${@c}"

@a = 10
@b = 20
@c = @a != @b
println "${@a} != ${@b} = ${@c}"
@a = 10
@b = 10
@c = @a != @b
println "${@a} != ${@b} = ${@c}"
@a = 20
@b = 10
@c = @a != @b
println "${@a} != ${@b} = ${@c}"

@a = 10
@b = 20
@c = @a == @b
println "${@a} == ${@b} = ${@c}"
@a = 10
@b = 10
@c = @a == @b
println "${@a} == ${@b} = ${@c}"
@a = 20
@b = 10
@c = @a == @b
println "${@a} == ${@b} = ${@c}"

##################################################

println "\n"
println "+--------------+"
println "| Conditionals |"
println "+--------------+"

##################################################

@a = true
@b = true

if @a == @b
    println "foo"
elsif @a != @b
    println "bar"
else
    println "baz"
endif

@a = true
@b = false

if @a == @b
    println "foo"
elsif @a != @b
    println "bar"
else
    println "baz"
endif

@a = true
@b = false

if @a == @b
    println "foo"
elsif @b == @a
    println "bar"
else
    println "baz"
endif

@a = true
@b = true

##################################################

println "\n"
println "+---------------------+"
println "| Nested Conditionals |"
println "+---------------------+"

##################################################

if @a == @b
    # should print
    println "hello world"

    if @a != @b
        println "foo"
    elsif @b == @a
        # should print
        println "bar"
    else
        println "baz"
    endif
elsif @a != @b
    println "hello"
else
    println "world"
endif
```