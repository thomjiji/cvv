import hashlib
import hmac
import textwrap


def generate_license_key(machine_id: str, secret_key: str = "secret") -> str:
    """
    machine_id: 你的机器 ID，比如 IOPlatformUUID
    secret_key: 服务器端保密的字符串（任何你自己设定的固定值）
    """
    # 用 HMAC-SHA256 把 machine_id 混合 secret_key
    digest = hmac.new(
        secret_key.encode("utf-8"), machine_id.encode("utf-8"), hashlib.sha256
    ).digest()
    print(digest.hex())
    # 取前 10 个字节（80 bits）作为简短 key
    short_bytes = digest[:10]
    # 转为大写十六进制
    hex_str = short_bytes.hex().upper()
    # 每 4 个字符一组，用 "-" 分隔
    grouped = "-".join(textwrap.wrap(hex_str, 4))
    return grouped


# 示例
machine_id = "4EACCC8E-237E-5459-A784-89BC3D60D3CC"  # 你的 IOPlatformUUID
key = generate_license_key(machine_id)
print(key)
