open Torch

module PMP = struct
  type t = {
    theta: Tensor.t;
    phi: Tensor.t -> Tensor.t;
    c: float;
    t: int;
  }

  let create phi c t =
    let theta = Tensor.randn [1] ~dtype:(T Float) in
    { theta; phi; c; t }

  let sample pmp x =
    let epsilon = Tensor.(neg (scalar pmp.c)) + Tensor.randn [1] ~dtype:(T Float) in
    let m_i_to_a = Tensor.zeros [1] ~dtype:(T Float) in
    let m_a_to_i = Tensor.zeros [1] ~dtype:(T Float) in
    
    let rec loop t m_i_to_a m_a_to_i =
      if t = pmp.t then
        let xi = Tensor.(argmax (pmp.phi x + epsilon + m_i_to_a) ~dim:0 ~keepdim:true) in
        xi
      else
        let m_i_to_a' = Tensor.(pmp.phi x + epsilon + m_i_to_a) in
        let m_a_to_i' = Tensor.((m_a_to_i * scalar 0.5) + (scalar 0.5 * (max m_i_to_a' ~dim:0 ~keepdim:true |> fst))) in
        loop (t + 1) m_i_to_a' m_a_to_i'
    in
    loop 0 m_i_to_a m_a_to_i

  let update pmp y_plus y_minus eta =
    let gradient = Tensor.((pmp.phi y_plus - pmp.phi y_minus) * scalar eta) in
    { pmp with theta = Tensor.(pmp.theta + gradient) }
end