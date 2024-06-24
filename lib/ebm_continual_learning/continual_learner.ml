open Torch

module ContinualLearner : ContinualLearnerSig = struct
  type t = {
    mask_dict : Tensor.t option ref;
    excit_buffer_list : Tensor.t list ref;
    si_c : float ref;
    epsilon : float ref;
    ewc_lambda : float ref;
    gamma : float ref;
    online : bool ref;
    fisher_n : int option ref;
    emp_FI : bool ref;
    ewc_task_count : int ref;
  }

  let create () =
    {
      mask_dict = ref None;
      excit_buffer_list = ref [];
      si_c = ref 0.;
      epsilon = ref 0.1;
      ewc_lambda = ref 0.;
      gamma = ref 1.;
      online = ref true;
      fisher_n = ref None;
      emp_FI = ref false;
      ewc_task_count = ref 0;
    }

  let _device self = Tensor.device @@ List.hd @@ Module.parameters self

  let _is_on_cuda self = Tensor.is_cuda @@ List.hd @@ Module.parameters self

  let forward x = failwith "Method 'forward' must be implemented by subclasses"

  let estimate_fisher self args dataset task allowed_classes collate_fn =
    let est_fisher_info = Hashtbl.create (module String) in
    Module.eval self;
    let data_loader =
      Utils.get_data_loader dataset ~batch_size:(Args.batch args)
        ~cuda:(self |> _is_on_cuda) ~collate_fn
    in
    let count = ref 0 in
    let rec loop index =
      match data_loader () with
      | [] -> ()
      | (x, y) :: tl ->
          incr count;
          if Option.value ~default:0 self.fisher_n <= index then ()
          else (
            let x = Tensor.to_device x @@ _device self in
            let y_hat = forward self x in
            let over_seen_classes = true in
            let y_tem =
              if over_seen_classes then
                let seen_classes_list =
                  List.init task ~f:(fun i -> self.labels_per_task.(i))
                in
                List.map (fun y' -> List.findi_exn (fun i c -> c = y') seen_classes_list) y
              else y
            in
            let negloglikelihood =
              F.nll_loss
                (F.log_softmax y_hat ~dim:(-1))
                (Tensor.to_device (Tensor.of_int_list y_tem) @@ _device self)
            in
            Module.zero_grad self;
            Tensor.backward negloglikelihood;
            let parameters = Module.named_parameters self in
            List.iter parameters ~f:(fun (n, p) ->
                if Tensor.requires_grad p then (
                  let n = String.substr_replace_all n ~pattern:"." ~with_:"__" in
                  let grad = Tensor.grad p in
                  let grad_sq = Tensor.pow grad 2.0 in
                  let est_fisher_entry =
                    match Hashtbl.find est_fisher_info n with
                    | Some entry -> entry + grad_sq
                    | None -> grad_sq
                  in
                  Hashtbl.set est_fisher_info ~key:n ~data:est_fisher_entry))
          );
          loop (index + 1)
    in
    loop 0;
    let index = Float.of_int !count in
    let est_fisher_info =
      Hashtbl.map est_fisher_info ~f:(fun fisher_entry ->
          Tensor.div fisher_entry index)
    in
    let parameters = Module.named_parameters self in
    List.iter parameters ~f:(fun (n, p) ->
        if Tensor.requires_grad p then (
          let n = String.substr_replace_all n ~pattern:"." ~with_:"__" in
          let prev_task_key =
            Printf.sprintf "%s_EWC_prev_task%s" n
              (if !online then "" else Int.to_string (!ewc_task_count + 1))
          in
          let fisher_key =
            Printf.sprintf "%s_EWC_estimated_fisher%s" n
              (if !online then "" else Int.to_string (!ewc_task_count + 1))
          in
          Module.register_buffer self prev_task_key (Tensor.detach p);
          if !online && !ewc_task_count = 1 then (
            let existing_values = Module.get_buffer self fisher_key in
            Hashtbl.set est_fisher_info ~key:n
              ~data:(Tensor.add existing_values (Tensor.mul !gamma fisher_entry))
          );
          Module.register_buffer self fisher_key fisher_entry
        ));
    if not !online then ewc_task_count := !ewc_task_count + 1;
    Module.train self ~mode:true
end