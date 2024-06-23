open Base
open Torch
open Torch.Nn
open Torch.Optim

type config = {
  queue_size : int;
  net : net_config;
  t : t_config;
  optim : optim_config;
  bs : int;
  momentum : float;
  temperature : float;
  p_iter : int;
  s_epoch : int;
  its : int;
}

and net_config = {
  proj_dim : int;
}

and t_config = { (* *) }

and optim_config = { (* *) }

let get_t size t_config = Torch.randn size |> Torch.cuda

let get_lr_schedule config it = Torch.FloatTensor.of_float (1.0)

let get_optimizer optim_config params = Torch_sgd.create params ~learning_rate:0.1

let load_ckpt enc_q proj_q enc_k proj_k queue opt path = (0, 0.0)

let save_ckpt enc_q proj_q enc_k proj_k queue opt it rt flag path = ()

let shuffled_idx n = 
  let shuffled_idxs = Torch.randperm n in
  let reverse_idxs = Torch.flip shuffled_idxs [0] in
  shuffled_idxs, reverse_idxs

let moco train_X enc_q proj_q enc_k proj_k config log_dir start_epoch =
  let train_n = Tensor.shape train_X |> Array.get 0 in
  let size = Tensor.shape train_X |> Array.get 2 in
  let nc = Tensor.shape train_X |> Array.get 1 in

  let mean = 
    Tensor.transpose train_X 0 1
    |> Tensor.flatten ~start_dim:1
    |> Tensor.mean ~dim:[1] ~keepdim:true
    |> Tensor.reshape ~shape:[|1; nc; 1; 1|]
    |> Tensor.cuda
  in

  let std = 
    Tensor.transpose train_X 0 1
    |> Tensor.flatten ~start_dim:1
    |> Tensor.mean ~dim:[1] ~keepdim:true
    |> Tensor.reshape ~shape:[|1; nc; 1; 1|]
    |> Tensor.cuda
  in

  let criterion = Torch_nn.Cross_entropy_loss.create () |> Torch.cuda in
  let queue = Tensor.randn [|config.queue_size; config.net.proj_dim|] |> Tensor.cuda in

  let params_q = List.concat [Enc_q.parameters enc_q; Proj_q.parameters proj_q] in
  let net_q = fun x -> x |> Tensor.sub mean |> Tensor.div std |> Enc_q.forward enc_q |> Proj_q.forward proj_q |> Tensor.normalize ~dim:1 in

  let params_k = List.concat [Enc_k.parameters enc_k; Proj_k.parameters proj_k] in
  let net_k = fun x -> x |> Tensor.sub mean |> Tensor.div std |> Enc_k.forward enc_k |> Proj_k.forward proj_k |> Tensor.normalize ~dim:1 in

  let t = get_t size config.t in
  let lr_schedule it = get_lr_schedule config it in
  let opt = get_optimizer config.optim params_q in

  let it, rt =
    if start_epoch = 0 then 0, 0.0
    else load_ckpt enc_q proj_q enc_k proj_k queue opt (log_dir ^ "/" ^ string_of_int start_epoch ^ ".pt")
  in

  Enc_q.train enc_q; Proj_q.train proj_q;
  Enc_k.train enc_k; Proj_k.train proj_k;

  let epoch_n_iter = Float.to_int (ceil ((Float.of_int train_n) /. (Float.of_int config.bs))) in
  let rec loop it rt =
    if it >= config.its then ()
    else (
      if it % epoch_n_iter = 0 then train_X |> Tensor.randperm |> Tensor.permute train_X;

      let i = it % epoch_n_iter in
      let it = it + 1 in

      let s = Unix.gettimeofday () in

      let x = Tensor.narrow train_X ~dim:0 ~start:(i * config.bs) ~length:config.bs in
      let x_v1 = x |> t |> Tensor.cuda in
      let x_v2 = x |> t |> Tensor.cuda in

      let z_v1 = x_v1 |> net_q |> Tensor.normalize ~dim:1 in

      let () =
        Tensor.no_grad (fun () ->
            List.iter2_exn (Enc_q.parameters enc_q) (Enc_k.parameters enc_k) ~f:(fun p_q p_k ->
                let p_k_data = p_k |> Tensor.mul (Torch.FloatTensor.create [|1.0 -. config.momentum|] [|1|] |> Tensor.cuda) in
                p_k_data |> Tensor.add_ (p_q |> Enc_q.parameters enc_q |> Tensor.detach |> Tensor.mul (Torch.FloatTensor.create [|config.momentum|] [|1|] |> Tensor.cuda))
              );

            List.iter2_exn (Proj_q.parameters proj_q) (Proj_k.parameters proj_k) ~f:(fun p_q p_k ->
                let p_k_data = p_k |> Tensor.mul (Torch.FloatTensor.create [|1.0 -. config.momentum|] [|1|] |> Tensor.cuda) in
                p_k_data |> Tensor.add_ (p_q |> Proj_q.parameters proj_q |> Tensor.detach |> Tensor.mul (Torch.FloatTensor.create [|config.momentum|] [|1|] |> Tensor.cuda))
              );

            let shuffled_idxs, reverse_idxs = shuffled_idx (Tensor.shape x_v2 |> Array.get 0) in
            let x_v2 = Tensor.index_select x_v2 ~dim:0 ~index:shuffled_idxs in
            let z_v2 = x_v2 |> net_k |> Tensor.normalize ~dim:1 in
            let z_v2 = z_v2 |> Tensor.index_select ~dim:0 ~index:reverse_idxs in
          )
      in

      let z_mem = queue |> Tensor.clone |> Tensor.detach |> Tensor.normalize ~dim:1 in

      let pos = Tensor.bmm (z_v1 |> Tensor.view ~size:[|Tensor.shape z_v1 |> Array.get 0; 1; -1|]) (z_v2 |> Tensor.view ~size:[|Tensor.shape z_v2 |> Array.get 0; -1; 1|]) |> Tensor.squeeze ~dim:[-1] in
      let neg = Tensor.mm z_v1 (z_mem |> Tensor.transpose ~dim0:0 ~dim1:1) in

      let logits = Tensor.cat ~dim:1 [|pos; neg|] |> Tensor.div_scalar ~scalar:config.temperature in
      let labels = Tensor.zeros_like logits |> Tensor.to_device Torch.cuda in
      let loss = criterion logits labels |> Tensor.to_float in

      opt |> Torch.set_lr (lr_schedule it);
      opt |> Torch.zero_grad;
      loss |> Tensor.backward;
      opt |> Torch.step;

      let queue = queue |> Tensor.narrow ~dim:0 ~start:(Tensor.shape z_v2 |> Array.get 0) ~length:(Tensor.shape queue |> Array.get 0 - Tensor.shape z_v2 |> Array.get 0) |> Tensor.cat ~dim:0 z_v2 |> Tensor.detach |> Tensor.clone in

      let e = Unix.gettimeofday () in
      let rt = rt +. (e -. s) in

      if it % config.p_iter = 0 then (
        save_ckpt enc_q proj_q enc_k proj_k queue opt it rt false (log_dir ^ "/curr.pt");
        Printf.printf "Epoch : %.3f | Loss : %.3f | LR : %.3e | Time : %.3f\n" (Float.of_int it /. Float.of_int epoch_n_iter) loss (lr_schedule it) rt
      );

      if it % (epoch_n_iter * config.s_epoch) = 0 then
        save_ckpt enc_q proj_q enc_k proj_k queue opt it rt true (Printf.sprintf "%s/%d.pt" log_dir (it / epoch_n_iter));

      loop it rt
    )
  in

  loop it rt