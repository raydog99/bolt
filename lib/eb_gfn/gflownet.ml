open Torch
open Base

type gflownet_randf_tb = {
  xdim : int;
  mutable hops : float;
  model : Torch.ScriptModule.t;
  logZ : Torch.tensor;
  device : Torch.Device.t;
  exp_temp : float;
  rand_coef : float;
  init_zero : bool;
  clip : float;
  l1loss : float;
  replay : 'a option;
  tau : int;
  train_steps : int;
  optimizer : Torch.Optim.t;
}

let make_mlp layers act with_bn =
  Torch.ScriptModule.load "path_to_your_script_module.pt"

let init_gflownet_randf_tb xdim args device net =
  let model =
    match net with
    | Some m -> m
    | None ->
        make_mlp
          ([| xdim |] @ Array.init args.hid_layers ~f:(fun _ -> args.hid))
          (if args.leaky then Torch.nn.leaky_relu else Torch.nn.relu)
          args.gfn_bn
  in
  let logZ = Torch.tensor [ 0. ] ~device in
  let param_list =
    [
      Torch.Optim.Params.create ~params:model#parameters ~lr:args.glr;
      Torch.Optim.Params.create ~params:logZ ~lr:args.zlr;
    ]
  in
  let optimizer =
    match args.opt with
    | "adam" ->
        Torch.Optim.adam param_list ~weight_decay:args.gfn_weight_decay
    | "sgd" ->
        Torch.Optim.sgd param_list ~momentum:args.momentum
          ~weight_decay:args.gfn_weight_decay
    | _ -> failwith "Unsupported optimizer"
  in
  {
    xdim;
    hops = 0.;
    model;
    logZ;
    device;
    exp_temp = args.temp;
    rand_coef = args.rand_coef;
    init_zero = args.init_zero;
    clip = args.clip;
    l1loss = args.l1loss;
    replay = None;
    tau = if args.tau then args.tau else -1;
    train_steps = args.train_steps;
    optimizer;
  }

  let backforth_sample t self x k rand_coef =
    assert (k > 0);
    let batch_size = Tensor.size x.(0) in
    let logp_xprime2x = Torch.zeros [ batch_size ] ~device:self.device in

    for step = 0 to k do
      let del_val_logits = self.model#forward x in
    done;

    let logp_x2xprime = Torch.zeros [ batch_size ] ~device:self.device in
    for step = 0 to (k - 1) do
      let logits = self.model#forward x in
    done;
    (x, logp_xprime2x - logp_x2xprime)

let sample self batch_size =
  let rec loop x step =
    if step < self.xdim then (
      let logits = self.model#forward x in
      let add_logits, _ = logits#slice 1 (2 * self.xdim) in
      loop x (step + 1)
    ) else x
  in
  let init_tensor =
    if self.init_zero then Torch.zeros [ batch_size; self.xdim ] ~device:self.device
    else -1. * Torch.ones [ batch_size; self.xdim ] ~device:self.device
  in
  loop init_tensor 0

let cal_logp self data num =
  let logp_ls = ref [] in
  for _ = 0 to num - 1 do
    let _, _, _, mle_loss =
      tb_mle_randf_loss (fun inp -> Torch.tensor [ 0. ] ~device:self.device) self data#size.(0) ~back_ratio:1 ~data
    in
    let logpj = Torch.neg mle_loss |> Torch.detach |> Torch.to_cpu |> Torch.neg |> Torch.log in
    logp_ls := logpj :: !logp_ls
  done;
  
  let logp_concat = Torch.cat !logp_ls ~dim:1 in
  let batch_logp = Torch.logsumexp logp_concat ~dim:1 |> Torch.mean in
  batch_logp

let evaluate self loader preprocess num use_tqdm =
  let logps = ref [] in
  let pbar =
    if use_tqdm then Torch.tqdm loader else loader
  in
  if Torch.hasattr pbar "set_description" then Torch.set_description pbar "Calculating likelihood";
  
  self.model#eval ();
  for (x, _) in pbar do
    let x = preprocess x |> Torch.to_device self.device in
    let logp = cal_logp self x num in
    logps := logp :: !logps;
    if Torch.hasattr pbar "set_postfix" then Torch.set_postfix pbar [ "logp", Printf.sprintf "%.2f" (Torch.mean (Torch.cat !logps) |> Torch.item) ];
  done;
  
  Torch.mean (Torch.cat !logps)

let train self batch_size scorer silent data back_ratio =
  let pbar =
    if silent then 0 -- self.train_steps
    else Torch.tqdm (0 -- self.train_steps)
  in
  let curr_lr = self.optimizer |> Torch.param_groups |> List.hd |> Torch.get_lr in
  if not silent then Torch.set_description pbar (Printf.sprintf "Lr=%.1e" curr_lr);
  
  let train_loss = ref [] in
  let train_mle_loss = ref [] in
  let train_logZ = ref [] in
  
  self.model#train ();
  self.model#zero_grad ();
  Torch.cuda.empty_cache ();

  for _ = 0 to self.train_steps - 1 do
    let gfn_loss, forth_loss, back_loss, mle_loss =
      tb_mle_randf_loss scorer self batch_size ~back_ratio ~data
    in
    let gfn_loss, forth_loss, back_loss, mle_loss =
      gfn_loss |> Torch.mean, forth_loss |> Torch.mean, back_loss |> Torch.mean, mle_loss |> Torch.mean
    in
    let loss = gfn_loss in

    self.optimizer#zero_grad ();
    loss |> Torch.backward;
    if self.clip > 0. then Torch.nn.utils.clip_grad_norm_ self.model#parameters self.clip ~norm_type:"inf";
    self.optimizer#step ();

    train_loss := gfn_loss |> Torch.item :: !train_loss;
    train_mle_loss := mle_loss |> Torch.item :: !train_mle_loss;
    train_logZ := self.logZ |> Torch.item :: !train_logZ;

    if not silent then
      Torch.set_postfix pbar
        [
          "MLE", Printf.sprintf "%.2e" (mle_loss |> Torch.item);
          "GFN", Printf.sprintf "%.2e" (gfn_loss |> Torch.item);
          "Forth", Printf.sprintf "%.2e" (forth_loss |> Torch.item);
          "Back", Printf.sprintf "%.2e" (back_loss |> Torch.item);
          "LogZ", Printf.sprintf "%.2e" (self.logZ |> Torch.item);
        ];
  done;

  (!train_loss |> List.mean, !train_logZ |> List.mean)