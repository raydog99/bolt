open Torch

type vanilla_net = {
  conv1 : nn -> nn;
  leakyrelu1 : unit -> nn;
  conv2 : int -> int -> int -> int -> nn;
  leakyrelu2 : unit -> nn;
  conv3 : int -> int -> int -> int -> nn;
  leakyrelu3 : unit -> nn;
  conv4 : int -> int -> int -> int -> nn;
  leakyrelu4 : unit -> nn;
  conv5 : int -> int -> int -> int -> nn
}

let vanilla_net n_c n_f leak =
  let conv1 = nn#conv2d n_c n_f 3 1 1 in
  let leakyrelu1 = nn#leaky_relu ~negative_slope:leak in
  let conv2 = nn#conv2d n_f (n_f * 2) 4 2 1 in
  let leakyrelu2 = nn#leaky_relu ~negative_slope:leak in
  let conv3 = nn#conv2d (n_f * 2) (n_f * 4) 4 2 1 in
  let leakyrelu3 = nn#leaky_relu ~negative_slope:leak in
  let conv4 = nn#conv2d (n_f * 4) (n_f * 8) 4 2 1 in
  let leakyrelu4 = nn#leaky_relu ~negative_slope:leak in
  let conv5 = nn#conv2d (n_f * 8) 1 4 1 0 in
  {
    conv1;
    leakyrelu1;
    conv2;
    leakyrelu2;
    conv3;
    leakyrelu3;
    conv4;
    leakyrelu4;
    conv5
  }

let forward vn x =
  let open Tensor in
  let conv1_out = vn.conv1 x |> vn.leakyrelu1 () in
  let conv2_out = vn.conv2 conv1_out |> vn.leakyrelu2 () in
  let conv3_out = vn.conv3 conv2_out |> vn.leakyrelu3 () in
  let conv4_out = vn.conv4 conv3_out |> vn.leakyrelu4 () in
  vn.conv5 conv4_out |> squeeze