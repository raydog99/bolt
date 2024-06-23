open Base
open Torch

let eval_learnt_params theta theta_exp params dataframe =
  params.optim.save_fig <- false;

  let n_data = params.leo.n_data_test in
  let traj_err_trans_test = Tensor.zeros [n_data; 1] in
  let traj_err_rot_test = Tensor.zeros [n_data; 1] in

  for data_idx = 0 to n_data - 1 do
    let x_opt, data, _, _ = optimizer_soln theta params data_idx ~dataset_mode:"test" in
    let x_opt = Tensor.tensor_of_float_array x_opt ~requires_grad:true ~dtype:Dtype.Float32 ~device:Device.cuda in

    let x_exp = get_exp_traj data theta_exp params in

    let traj_err_trans, traj_err_rot = quant_metrics.traj_error ~xyh_est:(Tensor.to_cpu x_opt |> Tensor.to_float_array1_exn)
                                                                ~xyh_gt:(Tensor.to_cpu x_exp |> Tensor.to_float_array1_exn) in

    Tensor.set traj_err_trans_test [|data_idx; 0|] traj_err_trans;
    Tensor.set traj_err_rot_test [|data_idx; 0|] traj_err_rot;

    let offset = params.leo.test_idx_offset in
    let nsteps = params.optim.nsteps - 1 in

    DataFrame.set_float dataframe ~row:(offset + data_idx, nsteps) ~column:"test/err/tracking/trans" traj_err_trans;
    DataFrame.set_float dataframe ~row:(offset + data_idx, nsteps) ~column:"test/err/tracking/rot" traj_err_rot;
  done;

  (traj_err_trans_test, traj_err_rot_test, dataframe)

let add_tracking_errors_to_dataframe df x_opt x_exp params =
  let nsteps =
    if params.dataio.dataset_type = "push2d" then
      int_of_float (0.5 *. float_of_int (Tensor.shape x_opt |> List.hd_exn))
    else
      Tensor.shape x_opt |> List.hd_exn
  in

  let x_opt_np = Tensor.to_cpu x_opt |> Tensor.to_float_array2_exn in
  let x_exp_np = Tensor.to_cpu x_exp |> Tensor.to_float_array2_exn in

  for tstep = 1 to nsteps - 1 do
    let err_trans, err_rot = quant_metrics.traj_error ~xyh_est:(Array.sub x_opt_np ~pos:0 ~len:tstep)
                                                     ~xyh_gt:(Array.sub x_exp_np ~pos:0 ~len:tstep) in
    DataFrame.set_float df ~row:tstep ~column:"train/err/tracking/trans" err_trans;
    DataFrame.set_float df ~row:tstep ~column:"train/err/tracking/rot" err_rot;
  done;

  df

let check_traj_convergence traj_err_trans traj_err_rot traj_err_trans_prev traj_err_rot_prev params =
  let diff_traj_err_trans = Float.abs (traj_err_trans -. traj_err_trans_prev) in
  let diff_traj_err_rot = Float.abs (traj_err_rot -. traj_err_rot_prev) in

  let eps_diff_traj_err_trans = params.leo.eps_diff_traj_err_trans in
  let eps_diff_traj_err_rot = params.leo.eps_diff_traj_err_rot in

  if diff_traj_err_trans < eps_diff_traj_err_trans && diff_traj_err_rot < eps_diff_traj_err_rot then
    true
  else
    false