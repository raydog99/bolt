open Base

let score_entropy data =
  let n = Array.length data in
  let d = Array.length data.(0) in
  let log_sum_exp xs =
    let max_x = Array.fold_left max (-infinity) xs in
    let sum = ref 0.0 in
    Array.iter (fun x -> sum := !sum +. exp (x -. max_x)) xs;
    max_x +. log !sum
  in
  let kl a = a *. log a -. a +. 1.0 in
  let rec hamming x y =
    if Array.length x = 0 then 0
    else if x.(0) = y.(0) then hamming (Array.sub x 1 (Array.length x - 1)) (Array.sub y 1 (Array.length y - 1))
    else 1 + hamming (Array.sub x 1 (Array.length x - 1)) (Array.sub y 1 (Array.length y - 1))
  in
  let rec score_sum x =
    if d = 0 then 0.0
    else
      let wx = log_sum_exp (Array.map (fun y -> if hamming x y = 1 then exp (score_network x y) else neg_infinity) data) in
      let wy = log_sum_exp (Array.map (fun y -> if hamming x y = 1 then score_network y x else neg_infinity) data) in
      (wx -. wy) +. kl (exp wy) +. score_sum (Array.sub x 1 (Array.length x - 1))
  in
  let loss = ref 0.0 in
  Array.iter (fun x -> loss := !loss +. score_sum x) data;
  !loss /. (float_of_int n)