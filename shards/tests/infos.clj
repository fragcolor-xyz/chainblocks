; SPDX-License-Identifier: BSD-3-Clause
; Copyright © 2021 Fragcolor Pte. Ltd.

; strings are compressed by default, need to unpack if we use info
(decompress-strings)
(prn (map (fn* [name] [name (shard-info name)]) (shards)))
(prn (map (fn* [name] [name (enum-info name)]) (enums)))