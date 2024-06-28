open Torch

module GaussianDistribution = struct
  let log_pdf mu sigma x =
    let z = Tensor.((x - mu) / sigma) in
    Tensor.(
      neg (
        log (mul (Float 2.) (Float Stdlib.Float.pi)) / (Float 2.) +
        log sigma +
        mul z z / (Float 2.)
      )
    )

  let sample mu sigma size =
    let eps = Tensor.randn size in
    Tensor.(mu + (sigma * eps))
end

module KLContraction = struct
  let objective p q x =
    let log_p = p x in
    let log_q = q x in
    Tensor.(mean (log_p - log_q))

  let gaussian_kl_contraction mu0 sigma0 sigma1 =
    let p = GaussianDistribution.log_pdf mu0 sigma0 in
    fun mu sigma ->
      let q = GaussianDistribution.log_pdf mu sigma in
      fun x ->
        let kl_pq = objective p q x in
        let t_x = GaussianDistribution.sample x sigma1 (Tensor.shape x) in
        let p_t = GaussianDistribution.log_pdf t_x sigma1 in
        let q_t = GaussianDistribution.log_pdf t_x sigma1 in
        let kl_pt_qt = objective p_t q_t t_x in
        Tensor.(kl_pq - kl_pt_qt)
end

module Optimization = struct
  let argmin f init_params lr max_iter =
    let rec loop params iter =
      if iter >= max_iter then params
      else
        let loss = f params in
        Torch.backward loss;
        let new_params = 
          List.map (fun p -> 
            Tensor.(sub p (mul (grad p) (Float lr)))
          ) params
        in
        List.iter Tensor.zero_grad params;
        loop new_params (iter + 1)
    in
    loop init_params 0
end