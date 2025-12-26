use candle_core::{Device, Result as CandleResult};
use std::path::Path;

pub fn check_cuda_available() -> bool {
    Device::new_cuda(0).is_ok()
}

pub fn create_cuda_device(device_id: usize) -> CandleResult<Device> {
    Device::new_cuda(device_id)
}

pub fn create_cpu_device() -> CandleResult<Device> {
    Ok(Device::Cpu)
}

pub fn get_device_info(device: &Device) -> String {
    match device {
        Device::Cpu => "CPU".to_string(),
        Device::Cuda(id) => format!("CUDA (Device {})", id),
        Device::Metal(_) => "Metal".to_string(),
    }
}

pub fn validate_model_path(path: &str) -> bool {
    Path::new(path).exists()
}

pub fn find_model_path(preferred: &[&str]) -> Option<String> {
    for path in preferred {
        if validate_model_path(path) {
            return Some(path.to_string());
        }
    }
    None
}
