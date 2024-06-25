open Torch

module EnergyModel = struct
  type t = {
    theta: Tensor.t;
    wx: Tensor.t;
    wa: Tensor.t;
  }

  let init () =
    let theta = Tensor.randn [1] in
    let wx = Tensor.randn [1] in
    let wa = Tensor.randn [1] in
    { theta; wx; wa }

  let energy_function x a w =
    Tensor.(x * w)

  let update_samples model x0 x1 a alpha k =
    let wx' = Tensor.(model.wx + (alpha / 2.) * grad (energy_function x0 a model.wx) + randn [1]) in
    let wa' = Tensor.(model.wa + (alpha / 2.) * grad (energy_function x1 a model.wa) + randn [1]) in
    { model with wx = wx'; wa = wa' }

  let update_x_a model x0 x1 a alpha k =
    let x' = Tensor.(x0 + (alpha / 2.) * grad (energy_function x0 a model.wx) + randn [1]) in
    let a' = Tensor.(a + (alpha / 2.) * grad (energy_function x0 a model.wa) + randn [1]) in
    x', a'

  let gradient_stopping x = x

  let loss_ml model x0 x1 a x_bar a_bar =
    let e1 = Tensor.(energy_function x1 a model.wx - energy_function x_bar a model.wx) in
    let e2 = Tensor.(energy_function x0 a model.wa - energy_function x0 a_bar model.wa) in
    Tensor.(relu e1 + relu e2)

  let loss_kl model x_bar a_bar =
    let e1 = energy_function x_bar a_bar model.wx in
    let e2 = energy_function x_bar a_bar model.wa in
    Tensor.(e1 + e2)

  let update model x0 x1 a x_bar a_bar learning_rate =
    let loss = Tensor.(loss_ml model x0 x1 a x_bar a_bar + loss_kl model x_bar a_bar) in
    let grad = Tensor.grad loss [model.theta] in
    let theta' = Tensor.(model.theta - learning_rate * (List.hd grad)) in
    { model with theta = theta' }
end

module Trainer = struct
  let train_step model x_train x_demo alpha k learning_rate =
    let x0, x1, a = x_demo in
    let model = EnergyModel.update_samples model x0 x1 a alpha k in
    let x0', x1', a' = x_train in
    let x_bar, a_bar = EnergyModel.update_x_a model x0' x1' a' alpha k in
    let x_bar = EnergyModel.gradient_stopping x_bar in
    let a_bar = EnergyModel.gradient_stopping a_bar in
    EnergyModel.update model x0' x1' a' x_bar a_bar learning_rate

  let train model x_train x_demo num_iterations alpha k learning_rate =
    let rec loop model i =
      if i = num_iterations then model
      else
        let model' = train_step model x_train x_demo alpha k learning_rate in
        loop model' (i + 1)
    in
    loop model 0
end