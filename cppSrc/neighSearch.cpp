#include <torch/extension.h>

#include <vector>

// s'(z) = (1 - s(z)) * s(z)
torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

// tanh'(z) = 1 - tanh^2(z)
torch::Tensor d_tanh(torch::Tensor z) {
  return 1 - z.tanh().pow(2);
}

// elu'(z) = relu'(z) + { alpha * exp(z) if (alpha * (exp(z) - 1)) < 0, else 0}
torch::Tensor d_elu(torch::Tensor z, torch::Scalar alpha = 1.0) {
  auto e = z.exp();
  auto mask = (alpha * (e - 1)) < 0;
  return (z > 0).type_as(z) + mask.type_as(z) * (alpha * e);
}

std::vector<torch::Tensor> lltm_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  auto X = torch::cat({old_h, input}, /*dim=*/1);

  auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));
  auto gates = gate_weights.chunk(3, /*dim=*/1);

  auto input_gate = torch::sigmoid(gates[0]);
  auto output_gate = torch::sigmoid(gates[1]);
  auto candidate_cell = torch::elu(gates[2], /*alpha=*/1.0);

  auto new_cell = old_cell + candidate_cell * input_gate;
  auto new_h = torch::tanh(new_cell) * output_gate;

  return {new_h,
          new_cell,
          input_gate,
          output_gate,
          candidate_cell,
          X,
          gate_weights};
}

std::vector<torch::Tensor> lltm_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate,
    torch::Tensor output_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights) {
  auto d_output_gate = torch::tanh(new_cell) * grad_h;
  auto d_tanh_new_cell = output_gate * grad_h;
  auto d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell;

  auto d_old_cell = d_new_cell;
  auto d_candidate_cell = input_gate * d_new_cell;
  auto d_input_gate = candidate_cell * d_new_cell;

  auto gates = gate_weights.chunk(3, /*dim=*/1);
  d_input_gate *= d_sigmoid(gates[0]);
  d_output_gate *= d_sigmoid(gates[1]);
  d_candidate_cell *= d_elu(gates[2]);

  auto d_gates =
      torch::cat({d_input_gate, d_output_gate, d_candidate_cell}, /*dim=*/1);

  auto d_weights = d_gates.t().mm(X);
  auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);

  auto d_X = d_gates.mm(weights);
  const auto state_size = grad_h.size(1);
  auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell};
}

std::vector<torch::Tensor> sortPointSet( torch::Tensor points, torch::Tensor supports){
  auto hMax = at::max(supports);
  // std::cout << "Output from pytorch module" << std::endl;
  // std::cout << "hMax " << hMax << std::endl;
  auto qMin = std::get<0>(at::min(points,0)) - hMax;
  auto qMax = std::get<0>(at::max(points,0)) + 2 * hMax;
  // std::cout << "qMin " << qMin << std::endl;
  // std::cout << "qMax " << qMax << std::endl;

  auto qEx = qMax - qMin;
  // std::cout << "qEx: " << qEx;
  
  auto cells = at::ceil(qEx / hMax).to(torch::kInt);
  // std::cout << "Cells: " << cells;
  auto indices = at::ceil((points - qMin) / hMax).to(torch::kInt);

  // auto linearIndices = at::empty({points.size(0)}, torch::TensorOptions().dtype(torch::kInt));

  auto linearIndices = indices.index({torch::indexing::Slice(), 0}) + cells[0] * indices.index({torch::indexing::Slice(), 1});
  // std::cout << __FILE__ << " " << __LINE__ << ": " << linearIndices << std::endl;

  auto indexAccessor = indices.accessor<int32_t, 2>();
  auto linearIndexAccessor = linearIndices.accessor<int32_t, 1>();
  auto cols = cells[0].item<int32_t>();
  int64_t batch_size = indices.size(0); 
  // at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
  //   for (int64_t b = start; b < end; b++) {
  //     linearIndexAccessor[b] = indexAccessor[b][0] + cols * indexAccessor[b][1];
  //     // linearIndices[b] = indices[b][0] + cells[0] * indices[b][1];
  //   }
  // });

  auto sorted = torch::argsort(linearIndices);
  // std::cout << __FILE__ << " " << __LINE__ << ": " << "" << std::endl;

  auto sortedIndices = torch::clone(linearIndices);
  auto sortedPositions = torch::clone(points);
  auto sortedSupport = torch::clone(supports);

  auto sort_ = sorted.accessor<int64_t, 1>();
  // std::cout << __FILE__ << " " << __LINE__ << ": " << "" << std::endl;
  auto sortedIndex_ = sortedIndices.accessor<int32_t, 1>();
  // std::cout << __FILE__ << " " << __LINE__ << ": " << "" << std::endl;
  auto sortedPosition_ = sortedPositions.accessor<float, 2>();
  // std::cout << __FILE__ << " " << __LINE__ << ": " << "" << std::endl;
  auto sortedSupport_ = sortedSupport.accessor<float, 1>();
  // std::cout << __FILE__ << " " << __LINE__ << ": " << "" << std::endl;
  auto points_ = points.accessor<float, 2>();
  // std::cout << __FILE__ << " " << __LINE__ << ": " << "" << std::endl;
  auto supports_ = supports.accessor<float,1>();
  // std::cout << __FILE__ << " " << __LINE__ << ": " << "" << std::endl;

  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    for (int64_t b = start; b < end; b++) {
      auto i = sort_[b];
      sortedIndex_[b] = linearIndexAccessor[i];
      sortedPosition_[b][0] = points_[i][0];
      sortedPosition_[b][1] = points_[i][1];
      sortedSupport_[b] = supports_[i];
    }
  });
  // auto b = 0;
  // std::cout << __FILE__ << " " << __LINE__ << ": " << sorted[b] << std::endl;
  // std::cout << __FILE__ << " " << __LINE__ << ": " << points[sort_[b]] << std::endl;
  // std::cout << __FILE__ << " " << __LINE__ << ": " << "" << std::endl;
  // sortedPositions[sorted] = points;

  // auto sortedIndices = linearIndices[sorted];
  // auto sortedPositions = points[sorted];
  // auto sortedSupport = supports[sorted];
  return {qMin, hMax, cells, sortedPositions, sortedSupport, sortedIndices, sorted};


  // std::cout << "indices: " << indices;


  torch::Tensor z_out = at::empty({points.size(0)}, points.options());

  return {z_out};
}

std::vector<torch::Tensor> buildNeighborList(
    torch::Tensor z) {

  torch::Tensor z_out = at::empty({z.size(0)}, z.options());
  int64_t batch_size = z.size(0); 

  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    for (int64_t b = start; b < end; b++) {
      z_out[b] = z[b] * z[b];
    }
  });

  return {z_out};
}


std::vector<torch::Tensor> createHashMap(
    torch::Tensor z) {

  torch::Tensor z_out = at::empty({z.size(0)}, z.options());
  int64_t batch_size = z.size(0); 

  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    for (int64_t b = start; b < end; b++) {
      z_out[b] = z[b] * z[b];
    }
  });

  return {z_out};
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lltm_forward, "LLTM forward");
  m.def("backward", &lltm_backward, "LLTM backward");
  m.def("buildNeighborList", &buildNeighborList, "LLTM backward (CUDA)");
  m.def("sortPointSet", &sortPointSet, "LLTM backward (CUDA)");
  m.def("createHashMap", &createHashMap, "LLTM backward (CUDA)");
}