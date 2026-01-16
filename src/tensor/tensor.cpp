#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    // Calculate strides
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    // Allocate storage
    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    // Return pointer to the data considering the offset
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    // Return number of dimensions
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    // Return shape of the tensor
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    // Return strides of the tensor
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    // Return data type of the tensor
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    // Return device type of the tensor
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    // Return device ID of the tensor
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    // Return number of elements in the tensor
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    // Return size of each element in bytes
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    size_t ndim = this->ndim();
    if (ndim == 0) {
        return true;  // Scalar tensor is always contiguous
    }

    const auto &shape = this->shape();
    const auto &strides = this->strides();

    // Check if stride[ndim-1] == 1 (last dimension stride must be 1)
    if (strides[ndim - 1] != 1) {
        return false;
    }

    // Check if strides[i] == strides[i+1] * shape[i+1] for all i < ndim-1
    for (size_t i = 0; i < ndim - 1; i++) {
        auto tmp = static_cast<ptrdiff_t>(strides[i + 1] * shape[i + 1]);
        if (strides[i] != tmp) {
            return false;
        }
    }

    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    
    if (order.size() != this->ndim()) {
        return nullptr; // TODO: throw exception
    }

    TensorMeta new_meta;
    new_meta.dtype = this->dtype();
    new_meta.shape.resize(this->ndim());
    new_meta.strides.resize(this->ndim());

    for (size_t i = 0; i < this->ndim(); i++) {
        new_meta.shape[i] = this->shape()[order[i]];
        new_meta.strides[i] = this->strides()[order[i]];
    }

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    auto total_elems = this->numel();
    size_t new_total_elems = 1;;
    for (auto s : shape) {
        new_total_elems *= s;
    }
    if (total_elems != new_total_elems) {
        return nullptr; // TODO: throw exception
    }

    if (!this->isContiguous()) {
        return nullptr; // TODO: throw exception
    }

    TensorMeta new_meta;
    new_meta.dtype = this->dtype();
    new_meta.shape = shape;
    // Calculate new strides
    size_t ndim_ = shape.size();
    new_meta.strides.resize(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        new_meta.strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // Create a new tensor which slices the original tensor along the given dimension.
    // No data transfer, only update the offset to point to the sliced region.

    if (dim >= this->ndim()) {
        return nullptr; // TODO: throw exception
    }
    if (start >= end || end > this->shape()[dim]) {
        return nullptr; // TODO: throw exception
    }

    TensorMeta new_meta;
    new_meta.dtype = this->dtype();
    new_meta.shape = this->shape();
    new_meta.strides = this->strides();  // Strides remain unchanged
    new_meta.shape[dim] = end - start;   // Update the size of sliced dimension
    
    // Calculate the new offset based on the start index in the sliced dimension
    // Note: _offset is in bytes, so we need to multiply strides by element size
    size_t new_offset = this->_offset + start * this->strides()[dim] * this->elementSize();
    
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}

void Tensor::load(const void *src) {
    size_t total_bytes = this->numel() * this->elementSize();
    
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        // Copy from host to host
        core::context().runtime().api()->memcpy_sync(
            this->data(),
            src,
            total_bytes,
            LLAISYS_MEMCPY_H2H);
    } else {
        // Copy from host to device (e.g., GPU)
        core::context().runtime().api()->memcpy_sync(
            this->data(),
            src,
            total_bytes,
            LLAISYS_MEMCPY_H2D);
    }
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
