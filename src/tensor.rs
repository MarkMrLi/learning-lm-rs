use std::{slice, sync::Arc, vec};
use half::{f16, bf16}; // 添加对 f16 和 bf16 的支持
use std::any::{TypeId, Any};

pub struct Tensor<T> {
    data: Arc<Box<[T]>>,
    shape: Vec<usize>,
    offset: usize,
    length: usize,
}

impl<T: Copy + Clone + Default + 'static> Tensor<T> {
    pub fn new(data: Vec<T>, shape: &Vec<usize>) -> Self {
        let length = data.len();
        Tensor {
            data: Arc::new(data.into_boxed_slice().try_into().unwrap()),
            shape: shape.clone(),
            offset: 0,
            length: length,
        }
    }

    /// 将张量转换为目标精度
    pub fn to_f32(&self, to_type: &dyn Any) -> Tensor<f32> {
        let now_type_id = TypeId::of::<T>();
        let to_type_id = to_type.type_id();

        // 处理 f16 -> f32 的转换
        if now_type_id == TypeId::of::<f16>() && to_type_id == TypeId::of::<f32>() {
            // 安全：已通过TypeId检查确保类型匹配
            let data_f16 = unsafe {
                &*(self.data.as_ref() as &[T] as *const [T] as *const [f16])
            };
            let new_data = data_f16.iter()
                .map(|x| x.to_f32())
                .collect::<Vec<f32>>();

            return Tensor{
                data: Arc::new(new_data.into_boxed_slice().try_into().unwrap()),
                shape: self.shape.clone(),
                offset: self.offset.clone(),
                length: self.length,
            }
        }
        // 添加其他类型转换...
        
        panic!("Unsupported type conversion from {:?} to {:?}", 
            now_type_id, to_type_id);
    }
    /// 将张量转换为目标精度
    pub fn to_f16(&self, to_type: &dyn Any) -> Tensor<f16> {
        let now_type_id = TypeId::of::<T>();
        let to_type_id = to_type.type_id();

        // 处理 f32 -> f16 的转换
        if now_type_id == TypeId::of::<f32>() && to_type_id == TypeId::of::<f16>() {
            // 安全：已通过TypeId检查确保类型匹配
            let data_f32 = unsafe {
                &*(self.data.as_ref() as &[T] as *const [T] as *const [f32])
            };
            let new_data = data_f32.iter()
                .map(|x| f16::from_f32(*x))
                .collect::<Vec<f16>>();
            return Tensor{
                data: Arc::new(new_data.into_boxed_slice().try_into().unwrap()),
                shape: self.shape.clone(),
                offset: self.offset.clone(),
                length: self.length,
            }
        } 

        panic!("Unsupported type conversion from {:?} to {:?}", 
            now_type_id, to_type_id);
        
    }
    pub fn default(shape: &Vec<usize>) -> Self {
        let length = shape.iter().product();
        let data = vec![T::default(); length];
        Self::new(data, shape)
    }

    pub fn data(&self) -> &[T] {
        &self.data[self.offset..][..self.length]
    }

    pub unsafe fn data_mut(&mut self) -> &mut [T] {
        let ptr = self.data.as_ptr().add(self.offset) as *mut T;
        slice::from_raw_parts_mut(ptr, self.length)
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn size(&self) -> usize {
        self.length
    }

    pub fn reshape(&mut self, new_shape: &Vec<usize>) -> &mut Self {
        let new_length: usize = new_shape.iter().product();
        if new_length != self.length {
            let old_shape = self.shape.clone();
            panic!("New shape {new_shape:?} does not match tensor of {old_shape:?}");
        }
        self.shape = new_shape.clone();
        self
    }

    pub fn slice(&self, start: usize, shape: &Vec<usize>) -> Self {
        let new_length: usize = shape.iter().product();
        assert!(self.offset + start + new_length <= self.length);
        Tensor {
            data: self.data.clone(),
            shape: shape.clone(),
            offset: self.offset + start,
            length: new_length,
        }
    }
}

// 测试和调试工具
impl Tensor<f32> {
    #[allow(unused)]
    pub fn close_to(&self, other: &Self, rel: f32) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        let a = self.data();
        let b = other.data();
        a.iter().zip(b).all(|(x, y)| float_eq(x, y, rel))
    }

    #[allow(unused)]
    pub fn print(&self) {
        println!(
            "shape: {:?}, offset: {}, length: {}",
            self.shape, self.offset, self.length
        );
        let dim = self.shape().last().cloned().unwrap_or_default();
        let batch = self.length / dim;
        for i in 0..batch {
            let start = i * dim;
            println!(
                "{:?}",
                &self.data()[start..].get(..dim).unwrap_or_default()
            );
        }
    }
}

#[inline]
pub fn float_eq(x: &f32, y: &f32, rel: f32) -> bool {
    (x - y).abs() <= rel * (x.abs() + y.abs()) / 2.0
}

// 其他数据类型实现
impl Tensor<f16> {
    #[allow(unused)]
    pub fn print(&self) {
        println!(
            "shape: {:?}, offset: {}, length: {}",
            self.shape, self.offset, self.length
        );
        let dim = self.shape().last().cloned().unwrap_or_default();
        let batch = self.length / dim;
        for i in 0..batch {
            let start = i * dim;
            println!(
                "{:?}",
                &self.data()[start..].get(..dim).unwrap_or_default()
            );
        }
    }
}

impl Tensor<bf16> {
    #[allow(unused)]
    pub fn print(&self) {
        println!(
            "shape: {:?}, offset: {}, length: {}",
            self.shape, self.offset, self.length
        );
        let dim = self.shape().last().cloned().unwrap_or_default();
        let batch = self.length / dim;
        for i in 0..batch {
            let start = i * dim;
            println!(
                "{:?}",
                &self.data()[start..].get(..dim).unwrap_or_default()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    pub fn test_transfor() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = Tensor::new(data, &shape);
        let tensor_f16 = tensor.to_f16(&f16::from_f32(0.0));
        for i in tensor_f16.data() {
            println!("{:?}", i);
        }
        let tensor_f32 = tensor_f16.to_f32(&f32::default());
        for i in tensor_f32.data() {
            println!("{:?}", i);
        }
        let a = f16::from_f32(1.3);
        println!("f16: {:?}", a);
        assert_eq!(2 + 2, 4);
    }
}