from app.services.acr_legacy.algorithms.slice_position import need_slice_pos, slice_pos_measure
from app.services.acr_legacy.algorithms.slice_thickness import need_slice_thickness, slice_thickness_measure
from app.services.acr_legacy.algorithms.geometric_accuracy import need_geometric, evaluate_geometric
from app.services.acr_legacy.algorithms.ghosting import need_ghosting, ghosting_measure
from app.services.acr_legacy.algorithms.piu import need_piu, piu_measure
from app.services.acr_legacy.algorithms.high_contrast_resolution import need_resolution, resolution_measure
from app.services.acr_legacy.algorithms.low_contrast_detectability import need_lcd, lcd_series_measure