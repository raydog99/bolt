open Torch

type resnet_block_conv1d = {
  size_in : int;
  size_h : int;
  size_out : int;
  bn_0 : nn;
  bn_1 : nn;
  fc_0 : nn;
  fc_1 : nn;
  fc_c : nn;
  actvn : nn;
  shortcut : nn option
}

let resnet_block_conv1d c_dim size_in ?(size_h=size_in) ?(size_out=size_in)
    norm_method legacy =
  let norm =
    match norm_method with
    | "batch_norm" -> nn#batch_norm1d size_in
    | "sync_batch_norm" -> nn#sync_batch_norm size_in
    | _ -> failwith ("Invalid norm method: " ^ norm_method)
  in
  let bn_0 = norm in
  let bn_1 = norm in
  let fc_0 = nn#conv1d size_in size_h 1 in
  let fc_1 = nn#conv1d size_h size_out 1 in
  let fc_c = nn#conv1d c_dim size_out 1 in
  let actvn = nn#relu in
  let shortcut =
    if size_in = size_out then None
    else Some (nn#conv1d size_in size_out 1)
  in
  {
    size_in;
    size_h;
    size_out;
    bn_0;
    bn_1;
    fc_0;
    fc_1;
    fc_c;
    actvn;
    shortcut
  }

let forward rbc x c =
  let open Tensor in
  let net = rbc.fc_0 (rbc.actvn (rbc.bn_0 x)) in
  let dx = rbc.fc_1 (rbc.actvn (rbc.bn_1 net)) in
  let x_s =
    match rbc.shortcut with
    | Some shortcut -> shortcut x
    | None -> x
  in
  let out = x_s + dx + rbc.fc_c c in
  out