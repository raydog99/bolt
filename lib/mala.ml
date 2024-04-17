open Base
open Torch

let sub = Var_store.sub

(* Metropolis-adjusted Langevin algorithm *)
let mala init_vals target_log_kernel draws_out vs =
	let n_burnin_draws = sub vs "n_burnin_draws" in
	let n_keep_draws = sub vs "n_keep_draws" in
	let n_total_draws = n_burnin_draws + n_keep_draws in
	let step_size = sub vs "step_size" in
	let vals_bound = sub vs "vals_bound" in
	let lower_bounds = sub vs "lower_bounds" in
	let upper_bounds = sub vs "upper_bounds"
;;