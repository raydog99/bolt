module Neuron : NeuronSig = struct
  let threshold = 20.0 (* Vthr *)
  let reset_potential = 0.0 (* Vreset *)
  let refractory_period = 2.0 (* tau_ref in ms *)
  let membrane_time_constant = 20.0 (* tau_m in ms *)

  let mutable potential = 0.0 (* V(t) *)
  let mutable synaptic_current = 0.0 (* I_syn *)
  let mutable refractory_time = 0.0

  let update_potential ~dt =
    if refractory_time > 0.0 then
      refractory_time <- refractory_time -. dt
    else
      let dv = ((-potential +. synaptic_current) /. membrane_time_constant) *. dt in
      potential <- potential +. dv

  let check_and_reset_spike () =
    if potential >= threshold then
      potential <- reset_potential;
      refractory_time <- refractory_period;
      true
    else
      false
end