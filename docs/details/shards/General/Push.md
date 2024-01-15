`Push` updates sequences and tables by pushing elements and/or sequences into them.

The name of the variable to update should come from the `Name:` parameter and the new update value(s) should come from the shard's input.

For existing sequences `Push` pushes in the new elements. If a sequence doesn't exist then `Push` will create it while pushing in the first element. These elements may be string constants, numerics, or even sequences themselves.

For tables `Push` can update only those existing keys whose values are of the type sequence. In such cases `Push` can push in new elements in those key-value pair sequences. The key to be updated must be passed in via the `Key:` parameter.

!!! note
   1. Do not use `Push` to update any variables created by `Set` (or its aliases `>=`/`>>=`). `Push` is best used to update variables that were themselves created by `Push` (first push).
   2. Though, if really want to do (1.) you can offload the current sequence into another sequence variable, push new values into it, and update the table with this sequence variable (see the last code example).  

The `Global:` parameter controls whether the created variables can be referenced across wires (`Global:` set to `true`) or only within the current wire (`Global:` set to `false`, default behaviour).

Variables may be locally scoped (created with `(Global: false)`; exists only for current wire) or globally scoped (created with `(Global: true)`; exists for all wires of that mesh). Hence, in update mode (i.e. when you apply `Push` to an existing variable) the `Global:` parameter is used in conjunction with the `Name:` parameter to identify the correct variable to update. 

The parameter `Clear:` controls whether we should clear out this sequence after every wire iteration (`Clear` set to `true`, default behaviour) or should the sequence data persist across wire iterations (`Clear` set to `false`).

The input to this shard is the new update value that is to be pushed into the sequence/table being modified. This value is also passed through as this shard's output.

!!! note
    `Push` has two aliases: `>>` which is an alias for `Push(... Clear: true)`, and `>>!` which is an alias for `Push(... Clear: false)`. See the code examples at the end to understand how these aliases are used.

!!! note "See also"
    - [`AppendTo`](../AppendTo)
    - [`PrependTo`](../PrependTo)
    - [`Sequence`](../Sequence)
    - [`Set`](../Set)
    - [`Table`](../Table)
    - [`Update`](../Update)
