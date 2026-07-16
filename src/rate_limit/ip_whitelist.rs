// Copyright (c) 2025-2026 Kirky.X
//
// Licensed under the MIT License
// See LICENSE file in the project root for full license information.

//! IP 白名单匹配 — 支持精确匹配与 IPv4/IPv6 CIDR 匹配。

/// 检查 IP 是否在白名单中（支持精确匹配与 CIDR 匹配）。
///
/// 白名单条目可以是：
/// - 精确 IP：`192.168.1.1`
/// - IPv4 CIDR：`192.168.0.0/16`
/// - IPv6 CIDR：`2001:db8::/32`
pub fn is_ip_whitelisted(ip: &str, whitelist: &[String]) -> bool {
    // 预解析 ip，避免在闭包内对每个 whitelist 条目重复 parse
    let ip_addr: std::net::IpAddr = match ip.parse() {
        Ok(addr) => addr,
        Err(_) => return whitelist.iter().any(|w| w == ip),
    };

    whitelist.iter().any(|whitelist_ip| {
        if ip == *whitelist_ip {
            return true;
        }

        if let Some((network_str, prefix_len_str)) = whitelist_ip.split_once('/')
            && let Ok(prefix_len) = prefix_len_str.parse::<u8>()
            && let Ok(network_addr) = network_str.parse::<std::net::IpAddr>()
        {
            return match (ip_addr, network_addr) {
                (std::net::IpAddr::V4(ip_v4), std::net::IpAddr::V4(network_v4)) => {
                    if prefix_len == 0 {
                        return true;
                    }
                    let ip_u32 = u32::from_be_bytes(ip_v4.octets());
                    let network_u32 = u32::from_be_bytes(network_v4.octets());
                    let mask = if prefix_len >= 32 {
                        0xFFFFFFFFu32
                    } else {
                        0xFFFFFFFFu32 << (32 - prefix_len)
                    };
                    (ip_u32 & mask) == (network_u32 & mask)
                }
                (std::net::IpAddr::V6(ip_v6), std::net::IpAddr::V6(network_v6)) => {
                    if prefix_len == 0 {
                        return true;
                    }
                    let ip_u128 = u128::from_be_bytes(ip_v6.octets());
                    let network_u128 = u128::from_be_bytes(network_v6.octets());
                    let mask = if prefix_len >= 128 {
                        u128::MAX
                    } else {
                        u128::MAX << (128u8 - prefix_len)
                    };
                    (ip_u128 & mask) == (network_u128 & mask)
                }
                _ => false,
            };
        }

        false
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match() {
        let whitelist = vec!["192.168.1.1".to_string()];
        assert!(is_ip_whitelisted("192.168.1.1", &whitelist));
        assert!(!is_ip_whitelisted("192.168.1.2", &whitelist));
    }

    #[test]
    fn test_ipv4_cidr_24() {
        let whitelist = vec!["192.168.1.0/24".to_string()];
        assert!(is_ip_whitelisted("192.168.1.100", &whitelist));
        assert!(!is_ip_whitelisted("192.168.2.1", &whitelist));
    }

    #[test]
    fn test_ipv4_cidr_0_matches_all() {
        let whitelist = vec!["0.0.0.0/0".to_string()];
        assert!(is_ip_whitelisted("1.2.3.4", &whitelist));
        assert!(is_ip_whitelisted("255.255.255.255", &whitelist));
    }

    #[test]
    fn test_ipv6_cidr_32() {
        let whitelist = vec!["2001:db8::/32".to_string()];
        assert!(is_ip_whitelisted("2001:db8:1234::1", &whitelist));
        assert!(!is_ip_whitelisted("2001:db9::1", &whitelist));
    }

    #[test]
    fn test_mismatched_ip_versions() {
        let whitelist = vec!["192.168.1.0/24".to_string()];
        assert!(!is_ip_whitelisted("2001:db8::1", &whitelist));
    }

    #[test]
    fn test_empty_whitelist() {
        let whitelist: Vec<String> = vec![];
        assert!(!is_ip_whitelisted("1.2.3.4", &whitelist));
    }

    #[test]
    fn test_invalid_ip_ignored() {
        let whitelist = vec!["192.168.1.0/24".to_string()];
        assert!(!is_ip_whitelisted("not-an-ip", &whitelist));
    }

    #[test]
    fn test_ipv4_cidr_32() {
        let whitelist = vec!["10.0.0.1/32".to_string()];
        assert!(is_ip_whitelisted("10.0.0.1", &whitelist));
        assert!(!is_ip_whitelisted("10.0.0.2", &whitelist));
    }
}
