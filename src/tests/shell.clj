; SPDX-License-Identifier: BSD-3-Clause
; Copyright © 2020 Fragcolor Pte. Ltd.

(def! n (Mesh))

(schedule
 n
 (Wire
  "n"
  "" (Process.Run "echo" ["Hello world"]) (Log)
  (Assert.Is "Hello world\n" true)
  "" (Process.Run "echo" ["Hello world"]) (Log)
  (Assert.Is "Hello world\n" true)

  ["10"] = .args
  (Maybe (-> "" (Process.Run "sleep" .args :Timeout 1) (Log)))))

;; (def! dec (fn* [a] (- a 1)))
;; (def! Loop (fn* [count] (do
;;   (if (tick n) nil (throw "tick failed"))
;;   (sleep 0.5)
;;   (if (> count 0) (Loop (dec count)) nil)
;; )))

;; (Loop 5)

(if (run n 0.1) nil (throw "Root tick failed"))