(* Integrate-and-fire neuron *)

module type NeuronSig = sig
  val mutable potential : float
  val threshold : float
  val reset_potential : float
  val refractory_period : float
  val membrane_time_constant : float
  val mutable synaptic_current : float
  val update_potential : dt:float -> unit
  val check_and_reset_spike : unit -> bool
end