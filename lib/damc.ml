open Torch
open Base

let sample_langevin_prior_z z netE vs = 
	let n_steps = Torch.sub vs "n_steps" in
	let s = Torch.sub vs "step_size" in
	let s = step_size in
	for i in range 1 to n_steps do
		let en = Tensor.sum (netE z) in
		let z_norm = 1.0 / 2.0 * Tensor.sum (z ** 2) in
		let z_grad = Tensor.grad (en + z_norm, z) |> fst in 
		let w = Tensor.randn_like z in
		z.data := z.data - 0.5 * (s ** 2) * z_grad + s * w;
	done
	let res = Tensor.detach z in
	{ res }

let sample_langevin_posterior_z z x netG netE vs =
	let n_steps = Torch.sub vs "n_steps" in
	let s = Torch.sub vs "step_size" in
	let sigma_inv = 1.0 / (2.0 * sigma ** 2) in
	for i in range 1 to n_steps do
		let x_hat = netG z in
		let g_log_lkhd = sigma_inv * Tensor.sum ( (x_hat - x) ** 2) in
		let z_n = 1.0 / 2.0 * Tensor.sum (z ** 2) in
		let e_n = Tensor.sum (netE z) in
		let total_en = g_log_lkhd + en + z_n in
		let z_grad = Tensor.grad (total_en, z) |> fst in 
		let w = Tensor.randn_like z in
		z.data := z.data - 0.5 * (s ** 2) * z_grad + s * w;
	done
	let res = Tensor.detach z in
	{ res }

(* Diffusion-amortized MCMC *)
let damc x netG netE Q_optimizer G_optimizer E_optimizer vs =
	let z_mask_prob = Tensor.randn len x in
	let z_mask = Tensor.ones len x in
	(* z_mask[z_mask_prob < 0.2] = 0.0; *)
	z_mask = Tensor.unsqueeze_last z_mask;

	(* Draw DAMC samples *)
	let z0 = Layer.forward Q x in
	let zk_pos = ref (Tensor.detach z0 |> Torch.clone) in
	let zk_neg = ref (Tensor.detach z0 |> Torch.clone) in

	(* Prior and posterior updates *)
	zk_pos := sample_langevin_posterior_z zk_pos x netG netE;
	zk_neg := sample_langevin_prior_z Torch.cat (Tensor.shape [zk_neg, Tensor.randn_like zk_neg], dim = 0) net E;

	(* Update Q *)
	for _ in range 6 do
		Optimizer.zero_grad Q_optimizer;
		let q_loss = Layer.forward Q calculate_loss x zk_pos z_mask |> mean in
		Tensor.backward q_loss;
		Optimizer Q_optimizer step;

	(* Update G *)
	Optimizer.zero_grad G_optimizer;
	let x_hat = Layer.forward G zk_pos in
	let g_loss = Tensor.sum (x_hat - x) ** 2 [1, 2, 3] |> mean in
	Tensor.backward g_loss;
	Optimizer.step G_optimizer;

	(* Update E *)
	Optimizer.zero_grad E_optimizer;
	let e_pos, e_neg = Layer.forward E zk_pos, Layer.forward E zk_neg in
	let e_loss = mean e_pos - mean e_neg in
	Tensor.backward e_loss;
	Optimizer.step E_optimizer;