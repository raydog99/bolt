open Torch

module InceptionNet = struct
  let inception_layers =
    let inception_v3 = Torch_vision.Models.inception_v3 ~pretrained:true ~transform_input:false () in
    Torch_nn.Sequential.of_list (Torch.Module.children inception_v3 |> List.tl)

  let forward x =
    let scale_factor = [|1; 1|] in
    let x = Tensor.to_type x Float in
    let x = Tensor.mul_scalar x 2.0 |> Tensor.add_scalar ~alpha:(-1.0) in
    let x = Torch_nn.forward inception_layers x in
    let x = Tensor.adaptive_avg_pool2d ~output_size:scale_factor x in
    Tensor.flatten ~start_dim:1 x
end