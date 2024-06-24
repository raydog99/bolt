open Torch

module type Discriminator = sig
  val forward : Tensor.t -> Tensor.t
  val log_density : Tensor.t -> Tensor.t
  val log_partition : unit -> Tensor.t
end

module type Generator = sig
  val log_density : Tensor.t -> Tensor.t
end

module CombinedDiscriminator = struct
  type t =
    { discriminator : (module Discriminator)
    ; generator : (module Generator) }

  let create discriminator generator =
    { discriminator; generator }

  let forward t x =
    let disc_mod = (val t.discriminator : Discriminator) in
    let gen_mod = (val t.generator : Generator) in
    Tensor.add (Discriminator.forward disc_mod x) (Generator.log_density gen_mod x)

  let log_density t x =
    let disc_mod = (val t.discriminator : Discriminator) in
    let gen_mod = (val t.generator : Generator) in
    try Tensor.sub (Discriminator.log_density disc_mod x) (Generator.log_density gen_mod x)
    with _ -> Tensor.sum (Tensor.zeros_like x) ~dim:[1]

  let log_partition t =
    let disc_mod = (val t.discriminator : Discriminator) in
    try Discriminator.log_partition disc_mod ()
    with _ -> Tensor.scalar 0.
end

module Identity = struct
  type t =
    { max_val : float }

  let create max_val =
    { max_val }

  let forward t x =
    x

  let inverse t x =
    x

  let log_grad t x =
    Tensor.zeros_like x
end