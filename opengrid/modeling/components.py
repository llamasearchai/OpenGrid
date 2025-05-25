"""
Power system component definitions for OpenGrid.

Author: Nik Jois <nikjois@llamasearch.ai>
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ComponentBase:
    """Base class for all power system components."""
    name: str
    component_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert component to dictionary."""
        return {
            "name": self.name,
            "type": self.component_type,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class Bus(ComponentBase):
    """Bus component for power systems."""
    vn_kv: float = 0.4
    zone: Optional[str] = None
    geodata: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        self.component_type = "bus"
        self.properties.update({
            "vn_kv": self.vn_kv,
            "zone": self.zone,
            "geodata": self.geodata
        })


@dataclass
class Line(ComponentBase):
    """Line component for power systems."""
    from_bus: int
    to_bus: int
    length_km: float
    std_type: str = "NAYY 4x50 SE"
    max_i_ka: Optional[float] = None
    r_ohm_per_km: Optional[float] = None
    x_ohm_per_km: Optional[float] = None
    c_nf_per_km: Optional[float] = None
    
    def __post_init__(self):
        self.component_type = "line"
        self.properties.update({
            "from_bus": self.from_bus,
            "to_bus": self.to_bus,
            "length_km": self.length_km,
            "std_type": self.std_type,
            "max_i_ka": self.max_i_ka,
            "r_ohm_per_km": self.r_ohm_per_km,
            "x_ohm_per_km": self.x_ohm_per_km,
            "c_nf_per_km": self.c_nf_per_km
        })


@dataclass
class Transformer(ComponentBase):
    """Transformer component for power systems."""
    hv_bus: int
    lv_bus: int
    std_type: str = "0.25 MVA 20/0.4 kV"
    sn_mva: Optional[float] = None
    vn_hv_kv: Optional[float] = None
    vn_lv_kv: Optional[float] = None
    vk_percent: Optional[float] = None
    vkr_percent: Optional[float] = None
    pfe_kw: Optional[float] = None
    i0_percent: Optional[float] = None
    
    def __post_init__(self):
        self.component_type = "transformer"
        self.properties.update({
            "hv_bus": self.hv_bus,
            "lv_bus": self.lv_bus,
            "std_type": self.std_type,
            "sn_mva": self.sn_mva,
            "vn_hv_kv": self.vn_hv_kv,
            "vn_lv_kv": self.vn_lv_kv,
            "vk_percent": self.vk_percent,
            "vkr_percent": self.vkr_percent,
            "pfe_kw": self.pfe_kw,
            "i0_percent": self.i0_percent
        })


@dataclass
class Load(ComponentBase):
    """Load component for power systems."""
    bus: int
    p_mw: float
    q_mvar: float = 0.0
    const_z_percent: float = 0.0
    const_i_percent: float = 0.0
    sn_mva: Optional[float] = None
    scaling: float = 1.0
    in_service: bool = True
    load_type: str = "residential"
    
    def __post_init__(self):
        self.component_type = "load"
        self.properties.update({
            "bus": self.bus,
            "p_mw": self.p_mw,
            "q_mvar": self.q_mvar,
            "const_z_percent": self.const_z_percent,
            "const_i_percent": self.const_i_percent,
            "sn_mva": self.sn_mva,
            "scaling": self.scaling,
            "in_service": self.in_service,
            "load_type": self.load_type
        })


@dataclass
class Generator(ComponentBase):
    """Generator component for power systems."""
    bus: int
    p_mw: float
    vm_pu: float = 1.0
    sn_mva: Optional[float] = None
    min_q_mvar: Optional[float] = None
    max_q_mvar: Optional[float] = None
    min_p_mw: float = 0.0
    max_p_mw: Optional[float] = None
    scaling: float = 1.0
    slack: bool = False
    controllable: bool = True
    generator_type: str = "conventional"
    fuel_type: str = "natural_gas"
    efficiency: float = 0.85
    
    def __post_init__(self):
        self.component_type = "generator"
        self.properties.update({
            "bus": self.bus,
            "p_mw": self.p_mw,
            "vm_pu": self.vm_pu,
            "sn_mva": self.sn_mva,
            "min_q_mvar": self.min_q_mvar,
            "max_q_mvar": self.max_q_mvar,
            "min_p_mw": self.min_p_mw,
            "max_p_mw": self.max_p_mw,
            "scaling": self.scaling,
            "slack": self.slack,
            "controllable": self.controllable,
            "generator_type": self.generator_type,
            "fuel_type": self.fuel_type,
            "efficiency": self.efficiency
        })


@dataclass 
class Switch(ComponentBase):
    """Switch component for power systems."""
    bus: int
    element: int
    et: str = "b"  # element type: 'b' for bus, 'l' for line
    closed: bool = True
    switch_type: str = "CB"  # Circuit Breaker
    z_ohm: float = 0.0
    
    def __post_init__(self):
        self.component_type = "switch"
        self.properties.update({
            "bus": self.bus,
            "element": self.element,
            "et": self.et,
            "closed": self.closed,
            "switch_type": self.switch_type,
            "z_ohm": self.z_ohm
        })


@dataclass
class Storage(ComponentBase):
    """Storage component for power systems."""
    bus: int
    p_mw: float = 0.0
    max_e_mwh: float = 1.0
    soc_percent: float = 50.0
    min_e_mwh: float = 0.0
    max_p_mw: Optional[float] = None
    min_p_mw: Optional[float] = None
    efficiency: float = 0.9
    storage_type: str = "battery"
    
    def __post_init__(self):
        self.component_type = "storage"
        self.properties.update({
            "bus": self.bus,
            "p_mw": self.p_mw,
            "max_e_mwh": self.max_e_mwh,
            "soc_percent": self.soc_percent,
            "min_e_mwh": self.min_e_mwh,
            "max_p_mw": self.max_p_mw,
            "min_p_mw": self.min_p_mw,
            "efficiency": self.efficiency,
            "storage_type": self.storage_type
        })


class ComponentFactory:
    """Factory for creating power system components."""
    
    _component_types = {
        "bus": Bus,
        "line": Line,
        "transformer": Transformer,
        "load": Load,
        "generator": Generator,
        "switch": Switch,
        "storage": Storage
    }
    
    @classmethod
    def create_component(cls, component_type: str, **kwargs) -> ComponentBase:
        """Create a component of the specified type."""
        if component_type not in cls._component_types:
            raise ValueError(f"Unknown component type: {component_type}")
        
        component_class = cls._component_types[component_type]
        component = component_class(**kwargs)
        
        logger.debug(
            "Component created",
            type=component_type,
            name=component.name,
            properties=component.properties
        )
        
        return component
    
    @classmethod
    def get_supported_types(cls) -> list[str]:
        """Get list of supported component types."""
        return list(cls._component_types.keys())


logger.info("OpenGrid component definitions loaded.") 