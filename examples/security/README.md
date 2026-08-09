# 安全模块示例

演示 VecBoost 安全基础设施 — 密钥管理、盐值生成和敏感数据脱敏。

## 示例

| 名称 | 说明 |
|------|------|
| `security_demo` | KeyStore 密钥存储、SaltStore 盐值、sanitize 脱敏函数 |

## 运行

```bash
cargo run -p vecboost-examples --bin security_demo
```

## 说明

安全模块提供以下能力：

### 密钥管理 (KeyStore)
- `SecretKey` — 密钥类型（JWT/API Key/DB Password/Model API Key）与掩码显示
- `EnvironmentKeyStore` — 基于环境变量的密钥存储
- `create_key_store(config)` — 根据配置创建密钥存储实例

### 盐值管理 (SaltStore)
- `SaltStore::generate()` — 生成 16 字节密码学随机盐
- `to_hex()` / `from_hex()` — hex 序列化往返

### 敏感数据脱敏
- `sanitize_secret(s)` — 显示首尾 2 字符，中间掩码
- `sanitize_password(s)` — 仅显示长度
- `sanitize_jwt_secret(s)` — 显示前缀 + 长度
