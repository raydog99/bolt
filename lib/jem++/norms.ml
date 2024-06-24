open Torch

module ConditionalInstanceNorm2dPlus = struct
  type t =
    { num_features : int
    ; bias : bool
    ; instance_norm : Layer.t
    ; embed : Layer.t }

  let create ~num_features ~num_classes ~bias =
    let instance_norm = Layer.instance_norm2d ~affine:false ~track_running_stats:false num_features in
    let embed =
      if bias then
        let embed = Layer.embedding ~num_embeddings:num_classes ~embedding_dim:(num_features * 3) in
        let _ = Tensor.normal_ (Layer.embedding_weight embed |> Tensor.narrow ~dim:1 ~start:0 ~length:(2 * num_features)) ~mean:1.0 ~std:0.02 in
        let _ = Tensor.zero_ (Layer.embedding_weight embed |> Tensor.narrow ~dim:1 ~start:(2 * num_features) ~length:num_features) in
        embed
      else
        let embed = Layer.embedding ~num_embeddings:num_classes ~embedding_dim:(2 * num_features) in
        let _ = Tensor.normal_ (Layer.embedding_weight embed) ~mean:1.0 ~std:0.02 in
        embed
    in
    { num_features; bias; instance_norm; embed }

  let forward t xs ys =
    let means = Tensor.mean xs ~dim:[|2; 3|] ~keepdim:false in
    let m = Tensor.mean means ~dim:[|-1|] ~keepdim:true in
    let v = Tensor.var means ~dim:[|-1|] ~keepdim:true in
    let means = Tensor.(means - m) / Tensor.(sqrt (v + f 1e-5)) in
    let h = Layer.forward t.instance_norm xs in
    let embed = Layer.forward t.embed ys in
    if t.bias then
      let gamma, rest = Tensor.split embed ~split_size:(t.num_features) ~dim:(-1) in
      let alpha, beta = Tensor.split rest ~split_size:(t.num_features) ~dim:(-1) in
      let h = Tensor.(h + means[..., None, None] * alpha[..., None, None]) in
      Tensor.(gamma.view ~size:[|-1; t.num_features; 1; 1|] * h + beta.view ~size:[|-1; t.num_features; 1; 1|])
    else
      let gamma, alpha = Tensor.split embed ~split_size:(t.num_features) ~dim:(-1) in
      let h = Tensor.(h + means[..., None, None] * alpha[..., None, None]) in
      Tensor.(gamma.view ~size:[|-1; t.num_features; 1; 1|] * h)
endd 