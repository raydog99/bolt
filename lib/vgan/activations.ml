open Torch

class type activation = object
  method apply : Tensor.t -> Tensor.t
end

class softmax : activation = object
  method apply x =
    let e_x = Tensor.exp (Tensor.sub x (Tensor.max2 x ~dim:1 ~keepdim:true)) in
    Tensor.div e_x (Tensor.sum e_x ~dim:1 ~keepdim:true)
end

class conv_softmax : activation = object
  method apply x =
    let e_x = Tensor.exp (Tensor.sub x (Tensor.max2 x ~dim:1 ~keepdim:true)) in
    Tensor.div e_x (Tensor.sum e_x ~dim:1 ~keepdim:true)
end

class maxout n_pool = object
  val n_pool = n_pool

  method apply x =
    let shape = Tensor.shape x in
    match shape with
    | [ batch_size; _ ] ->
      let slices = List.init n_pool (fun n -> Tensor.narrow x ~dim:1 ~start:n ~length:n_pool) in
      Tensor.max_list slices ~dim:0
    | [ batch_size; channels; height; width ] ->
      let slices = List.init n_pool (fun n -> Tensor.narrow x ~dim:1 ~start:n ~length:n_pool) in
      Tensor.max_list slices ~dim:0
    | _ -> failwith "NotImplemented"
end

class rectify : activation = object
  method apply x = Tensor.div (Tensor.add x (Tensor.abs x)) (Tensor.scalar_float 2.0)
end

class clipped_rectify clip = object
  val clip = clip

  method apply x =
    Tensor.clamp (Tensor.div (Tensor.add x (Tensor.abs x)) (Tensor.scalar_float 2.0)) ~min:0.0 ~max:clip
end

class leaky_rectify leak = object
  val leak = leak

  method apply x =
    let f1 = Tensor.scalar_float (0.5 *. (1.0 +. leak)) in
    let f2 = Tensor.scalar_float (0.5 *. (1.0 -. leak)) in
    Tensor.add (Tensor.mul f1 x) (Tensor.mul f2 (Tensor.abs x))
end

class prelu = object
  method apply x leak =
    let shape = Tensor.shape x in
    let leak = if List.length shape = 4 then Tensor.view leak ~size:[1; List.nth shape 1; 1; 1] else leak in
    let f1 = Tensor.scalar_float 0.5 |> Tensor.add leak in
    let f2 = Tensor.scalar_float 0.5 |> Tensor.sub leak in
    Tensor.add (Tensor.mul f1 x) (Tensor.mul f2 (Tensor.abs x))
end

class tanh : activation = object
  method apply x = Tensor.tanh x
end

class sigmoid : activation = object
  method apply x = Tensor.sigmoid x
end

class linear : activation = object
  method apply x = x
end

class hard_sigmoid : activation = object
  method apply x =
    Tensor.clamp (Tensor.add_scalar x 0.5) ~min:0.0 ~max:1.0
end

class trec t = object
  val t = t

  method apply x =
    Tensor.mul x (Tensor.gt_scalar x (Tensor.scalar_float t))
end

class hard_tanh : activation = object
  method apply x =
    Tensor.clamp x ~min:(Tensor.scalar_float (-1.0)) ~max:(Tensor.scalar_float 1.0)
end
