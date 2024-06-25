open Torch

let device = Device.cuda_if_available ()

let z_dim = 100
let h_dim = 128
let x_dim = 784
let lr = 0.0002

let xavier_init m =
  let fan_in = Tensor.shape m |> List.hd in
  let scale = sqrt (2.0 /. Float.of_int fan_in) in
  Tensor.mul_scalar (Tensor.randn_like m) scale

class gen = object (self)
  inherit nn.module_ as super

  val model =
    nn.Sequential.sequential [
      nn.Linear.linear ~input_dim:z_dim h_dim;
      nn.functional.relu;
      nn.Linear.linear ~input_dim:h_dim x_dim;
      nn.functional.sigmoid
    ]

  method forward input =
    nn.Module.forward_ model input
end

class dis = object (self)
  inherit nn.module_ as super

  val model =
    nn.Sequential.sequential [
      nn.Linear.linear ~input_dim:x_dim h_dim;
      nn.functional.relu;
      nn.Linear.linear ~input_dim:h_dim x_dim;
    ]

  method forward input =
    nn.Module.forward_ model input
end

let g = new gen
let d = new dis

let g = nn.Module.to_device g device
let d = nn.Module.to_device d device

let g_solver = Optim.Adam.create ~learning_rate:lr (nn.Module.parameters g)
let d_solver = Optim.Adam.create ~learning_rate:lr (nn.Module.parameters d)