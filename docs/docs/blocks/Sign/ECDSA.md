# Sign.ECDSA

```clojure
(Sign.ECDSA
  :Key [(Bytes) (ContextVar [(Bytes)]) (String) (ContextVar [(String)])]
)
```

## Definition


## Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| Key | `[(Bytes) (ContextVar [(Bytes)]) (String) (ContextVar [(String)])]` | `None` | The private key to be used to sign the hashed message input. |


## Input
| Type | Description |
|------|-------------|
| `[(Bytes)]` |  |


## Output
| Type | Description |
|------|-------------|
| `[(Seq [(Bytes)])]` |  |


## Examples

```clojure
(Sign.ECDSA

)
```
