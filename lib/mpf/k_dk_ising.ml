open Torch

module K_dK_ising = struct
  let forward j x =
    let ndims, nbatch = Tensor.(size x |> to_int2_exn) in
    let j = Tensor.reshape j [|ndims; ndims|] in
    let j = Tensor.((add j (transpose j ~dim0:0 ~dim1:1)) / (Float 2.)) in
    let y = Tensor.mm j x in
    let diag_j = Tensor.diag j in
    let x_not_x = Tensor.((mul x (Float 2.)) - (ones [ndims; nbatch])) in
    let k_full = Tensor.(exp (add (mul x_not_x y) (mul (Float (-0.5)) (repeat diag_j ~repeats:[|1; nbatch|])))) in
    let k = Tensor.sum k_full in
    
    let lt = Tensor.mul k_full x_not_x in
    let dj = Tensor.mm lt (Tensor.transpose x ~dim0:0 ~dim1:1) in
    let dj = Tensor.(sub dj (mul (Float 0.5) (diag (sum k_full ~dim:[1])))) in
    
    let dj = Tensor.((add dj (transpose dj ~dim0:0 ~dim1:1)) / (Float 2.)) in
    let dk = Tensor.flatten dj in
    
    let k = Tensor.(div k (Float (float_of_int nbatch))) in
    let dk = Tensor.(div dk (Float (float_of_int nbatch))) in
    
    k, dk
end