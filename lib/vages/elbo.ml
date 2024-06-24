open Torch

module type Criterion = sig
  type t
  val create : n_particles:int -> models:ModelsManager.t -> optimizers:OptimizersManager.t -> lr_schedulers:LRSchedulersManager.t -> t
  val criterion_name : t -> string
  val objective : t -> Tensor.t -> Tensor.t
end

module ELBO : Criterion = struct
  type t =
    { n_particles : int
    ; lvm : Layer.t
    ; q : Layer.t }

  let create ~n_particles ~models ~optimizers:_ ~lr_schedulers:_ =
    { n_particles; lvm = models.ModelsManager.lvm; q = models.ModelsManager.q }

  let criterion_name t =
    Printf.sprintf "ELBO%d" t.n_particles

  let objective t v =
    Tensor.(neg (vi_elbo v t.lvm t.q t.n_particles))
end

module IWAE : Criterion = struct
  type t =
    { n_particles : int
    ; lvm : Layer.t
    ; q : Layer.t }

  let create ~n_particles ~models ~optimizers:_ ~lr_schedulers:_ =
    { n_particles; lvm = models.ModelsManager.lvm; q = models.ModelsManager.q }

  let criterion_name t =
    Printf.sprintf "IWAE%d" t.n_particles

  let objective t v =
    Tensor.(neg (vi_iwae v t.lvm t.q t.n_particles))
end

let vi_elbo v lvm q n_particles =
  Tensor.f 0.0

let vi_iwae v lvm q n_particles =
  Tensor.f 0.0