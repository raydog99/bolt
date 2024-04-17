open Base
open Torch

let img_size, batch_size = 32, 100
let nz, nc, ndf, ngf = 100, 3, 200, 64
let k_0, a_0, k_1, a_1 = 60, 0.4, 40, 0.1
let llhd_sigma = 0.3
let n_iter = 70000

let create_generator vs =
	Layer.of_fn_ (fun xs -> 
		Layer.forward Layer.conv_transpose2d (nz) (ngf * 4) (4) (1) (0)
		|> Tensor.Leaky_relu
		|> Layer.forward Layer.conv_transpose2d (ngf * 8) (ngf * 4) (4) (2) (1)
		|> Tensor.Leaky_relu
		|> Layer.forward Layer.conv_transpose2d (ngf * 4) (ngf * 4) (4) (2) (1)
		|> Tensor.Leaky_relu
		|> Layer.forward Layer.conv_transpose2d (ngf * 2) (nc) (4) (2) (1)
		|> Tensor.Tanh
	)

let create_ebm vs =
	Layer.of_fn_ (fun xs z ->
		let z = z.squeeze in
		let view = (-1, 1, 1, 1) in
		Layer.forward Layer.linear (nz) (ndf)
		|> Tensor.Leaky_relu 0.2
		|> Layer.linear (ndf) (ndf)
		|> Tensor.Leaky_relu 0.2
		|> Layer.linear (ndf) (1)
	)

let transform () =
  Transform.compose [
    Resize [img_size]; ToTensor; Normalize ([0.5; 0.5; 0.5], [0.5; 0.5; 0.5])
  ]

let sample_p_data () =
  Tensor.index_select ~dim:0 data (Tensor.to_device (LongTensor.randn [|batch_size|]) device)

let sample_p_0 (n = batch_size) =
  Tensor.randn ~device [|n; nz; 1; 1|]

let sample_langevin_prior z e =
	let z = Tensor.clone z ~device ~requires_grad:true in
	for k in 1 to k_0 do
		let en = Layer.forward e z in
		let z_grad = Tensor.grad (Tensor.sum en) z |> fst in
		z.data = z.data - 0.5 * a_0 * a_0 * (z_grad + 1.0)/(z.data) + a_0 * Tensor.randn_like(z).data
	done
	Tensor.detach z

let sample_langevin_posterior z x g e =
	let z = Tensor.clone z ~device ~requires_grad:true in
	for k in 1 to k_1 do
		let x_hat = Layer.forward g z in
		let g_log_lkhd = 1.0 / (2.0 * llhd_sigma * llhd_sigma) * mse(x_hat, x) in
		let grad_g = Tensor.grad (g_log_lkhd z) |> fst in
		let en = Layer.forward e z in
		let grad_e = Tensor.grad (Tensor.sum en) |> fst in
		let z_data = Tensor.(z * (fmsub ((a_1 *. a_1) *. (grad_g + grad_e + (Tensor.ones_like z) / z)))) in
	    Tensor.(z .= (z_data - (a_1 *. randn_like z_data)))
	  done;
	  Tensor.detach z

let () =
	let device = Device.cuda_if_available () in
	let generator_vs = Var_store.create ~name:"gen" ~device () in
	let generator = create_generator generator_vs in
	let ebm_vs = Var_store.create ~name:"ebm" ~device () in
	let ebm = create_ebm ebm_vs in
	let mse = Torch_core.Reduction.Sum in
	let opt_e = Optimizer.Adam ebm_vs ~learning_rate:2e-5 ~beta1:0.5 ~beta2:0.999 in
	let ope_g = Optimizer.Adam generator_vs ~learning_rate: 1e-4 ~beta:0.5 ~beta2:0.999 in

	for i in range 1 to n_iter do
		let x = sample_p_data () in
		let z_e_0, z_g_0 = sample_p_0 (), sample_p_0 () in
		let z_e_k, z_g_k = sample_langevin_prior z_e_0 e, sample_langevin_prior z_g_0 x generator ebm in
		
		Optimizer.zero_grad opt_e;
		let x_hat = Layer.forward generator (Tensor.detach z_g_k) in
		let loss_g = mse x_hat x in
		Tensor.backward loss_g;
		Optimizer.step opt_g;

		Optimizer.zero_grad opt_g;
		let en_pos = Layer.forward ebm (Tensor.detach z_g_k) |> Tensor.mean in
		let en_neg = Layer.forward ebm (Tensor.detach z_e_k) |> Tensor.mean in
		let loss_e = en_pos - en_neg in
		Tensor.backward loss_e;
		Optimizer.step opt_e;