open Torch

type basic_block = {
  bn1 : nn -> nn;
  relu1 : unit -> nn;
  conv1 : int -> int -> int -> int -> nn;
  bn2 : nn -> nn;
  relu2 : unit -> nn;
  conv2 : int -> int -> int -> int -> nn;
  droprate : float;
  equalInOut : bool;
  convShortcut : nn -> nn
}

let basic_block in_planes out_planes stride ?(dropRate=0.0) () =
  let bn1 = nn#batch_norm2d in_planes in
  let relu1 = nn#relu ~inplace:true in
  let conv1 = nn#conv2d in_planes out_planes 3 stride 1 ~bias:false in
  let bn2 = nn#batch_norm2d out_planes in
  let relu2 = nn#relu ~inplace:true in
  let conv2 = nn#conv2d out_planes out_planes 3 1 1 ~bias:false in
  let equalInOut = in_planes = out_planes in
  let convShortcut =
    if not equalInOut then
      nn#conv2d in_planes out_planes 1 stride 0 ~bias:false
    else
      (fun _ -> nn)
  in
  {
    bn1;
    relu1;
    conv1;
    bn2;
    relu2;
    conv2;
    droprate = dropRate;
    equalInOut;
    convShortcut
  }

let forward bb x =
  let open Tensor in
  let relu1_bn1 = bb.relu1 () (bb.bn1 x) in
  let conv1_out =
    if bb.equalInOut then
      bb.conv1 relu1_bn1
    else
      bb.conv1 x
  in
  let relu2_bn2 = bb.relu2 () (bb.bn2 conv1_out) in
  let dropout_out =
    if bb.droprate > 0. then F.dropout relu2_bn2 ~p:bb.droprate ~training:true
    else relu2_bn2
  in
  let conv2_out = bb.conv2 dropout_out in
  if not bb.equalInOut then
    Tensor.add (bb.convShortcut x) conv2_out
  else
    Tensor.add x conv2_out