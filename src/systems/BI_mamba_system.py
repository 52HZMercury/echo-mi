from .mi_mamba_echo_prime_text_video_system import MIMambaEchoPrimeTextVideoSystem


class BIMambaSystem(MIMambaEchoPrimeTextVideoSystem):
    """Binary-classification training system shared by BI-Mamba variants."""


__all__ = ["BIMambaSystem"]
