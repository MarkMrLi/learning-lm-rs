use std::ops::{AddAssign, MulAssign};

use num_traits::Float;
use crate::tensor::{FloatConvert, Tensor};

// get (row) vectors from a 2D table given a list of indices
pub fn gather<T:Copy + Default + Clone + Float +'static>(y: &mut Tensor<T>, indices: &Tensor<u32>, table: &Tensor<T>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope<T: Default + Copy + Float + FloatConvert + 'static>(
    y: &mut Tensor<T>, 
    start_pos: usize, 
    theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i].clone().to_f32();
                let b = data[tok * n_heads * d + head * d + i + d / 2].clone().to_f32();
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = T::from_f32(a * cos - b * sin);
                data[tok * n_heads * d + head * d + i + d / 2] = T::from_f32(b * cos + a * sin);
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
// pub fn masked_softmax(y: &mut Tensor<f32>) {
//     todo!("精度敏感操作");
//     let ndim = y.shape().len();
//     assert!(ndim >= 2);
//     let seq_len = y.shape()[ndim - 2];
//     let total_seq_len = y.shape()[ndim - 1];
//     let batch = y.size() / (seq_len * total_seq_len);
//     let data = unsafe { y.data_mut() };
//     for b in 0..batch {
//         let base = b * seq_len * total_seq_len;
//         for i in 0..seq_len {
//             let offset = base + i * total_seq_len;
//             let boundary = total_seq_len - seq_len + i + 1;

//             let max = data[offset..offset + boundary]
//                 .iter()
//                 .fold(data[offset], |a, b| a.max(*b));

//             let sum = (0..boundary)
//                 .map(|j| {
//                     let e = (data[offset + j] - max).exp();
//                     data[offset + j] = e;
//                     e
//                 })
//                 .sum::<f32>();

//             (0..boundary).for_each(|j| data[offset + j] /= sum);
//             (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
//         }
//     }
// }

pub fn masked_softmax<T: FloatConvert + Copy + Default + 'static>(y: &mut Tensor<T>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    
    // 获取维度信息
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    
    // 获取数据指针并转换为目标类型
    let data = unsafe { y.data_mut() };

    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            // 阶段 1: 查找最大值（转换为 f32）
            let max = data[offset..offset + boundary]
                .iter()
                .map(|&v| v.to_f32())
                .fold(f32::NEG_INFINITY, |a, b| a.max(b));

            // 阶段 2: 计算指数和（保持 f32 精度）
            let sum = (0..boundary)
                .map(|j| {
                    let orig_val = data[offset + j].to_f32();
                    let e = (orig_val - max).exp();
                    e
                })
                .sum::<f32>();

            // 阶段 3: 写回结果（转换回原始类型）
            (0..boundary).for_each(|j| {
                let orig_val = data[offset + j].to_f32();
                let e = (orig_val - max).exp();
                data[offset + j] = T::from_f32(e / sum);
            });

            // 阶段 4: 处理 padding 部分
            (boundary..total_seq_len).for_each(|j| {
                data[offset + j] = T::from_f32(0.0);
            });
        }
    }
}
pub fn rms_norm<T: Copy + FloatConvert + Default + Clone + 'static>(
    y: &mut Tensor<T>,
    x: &Tensor<T>,
    w: &Tensor<T>,
    epsilon: f32,
) {
    // 检查形状一致性
    let dim = x.shape().last().unwrap();
    assert_eq!(w.size(), *dim, "w 的维度必须和 x 的维度相等");

    // 计算总向量数量
    let m = x.size() / dim;

    // 获取数据指针
    let x_data = x.data();
    let w_data = w.data();
    let y_data =unsafe {
        y.data_mut()
    }; 

    for i in 0..m {
        let idx_base = i * dim;
        
        // 计算平方和
        let mut sum_sq = 0.0;
        for j in 0..*dim {
            let val = x_data[idx_base + j].clone().to_f32();
            sum_sq += val.powi(2);
        }

        // 计算缩放因子
        let scale = 1.0 / (sum_sq / (*dim as f32) + epsilon).sqrt();

        // 计算结果并写回
        for j in 0..*dim {
            let x_val = x_data[idx_base + j].clone().to_f32();
            let w_val = w_data[j].clone().to_f32();
            let result = T::from_f32(w_val * x_val * scale);
            y_data[idx_base + j] = result;
        }
    }
}

// y = silu(x) * y
// hint: this is an element-wise operation
pub fn swiglu<T:Default + Copy + FloatConvert + Float + MulAssign + 'static>(y: &mut Tensor<T>, x: &Tensor<T>) {
    let len = y.size();
    assert!(len == x.size());

    let _y = unsafe { y.data_mut() };
    let _x = x.data();

    for i in 0..len {
        let x_val = _x[i];
        let sigmoid = T::from_f32(1.0) / (T::from_f32(1.0) + (-x_val).exp());
        _y[i] *= x_val * sigmoid;
    }

    // todo!("实现 silu，这里给了一些前期准备工作的提示，你可以参考")
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb<T: Default + Copy + Float + AddAssign + 'static>(
    c: &mut Tensor<T>, 
    beta: T, 
    a: &Tensor<T>, 
    b: &Tensor<T>, 
    alpha: T) {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape.first().unwrap();
    let n = b_shape.first().unwrap();
    
    assert_eq!(a_shape.last(), b_shape.last(), "假设 A B 均为二维张量");
    let k = a_shape.last().unwrap();
    
    let _c = unsafe { c.data_mut() };
    let _a = a.data();
    let _b = b.data();

    for i in 0..*m {
        for j in 0..*n {
            let mut sum = T::default();
            for l in 0..*k {
                sum += _a[i * *k + l] * _b[j * *k + l];
            }
            _c[i * *n + j] = beta * _c[i * *n + j] + alpha * sum;
        }
    }
    // todo!("实现 matmul_transb，计算前做一些必要的检查会帮助你后续调试");
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample<T:Default + Copy + Float + FloatConvert + 'static>(x: &Tensor<T>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl<T:FloatConvert + Copy> From<(usize, &T)> for Probability {
        #[inline]
        fn from((i, p): (usize, &T)) -> Self {
            Self {
                val: T::to_f32(*p),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    swiglu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}
