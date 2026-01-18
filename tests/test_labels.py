from core.sky_core import _select_labels


def test_label_clustering():
    """Verifica que si dos objetos están pegados, solo se etiquete uno."""
    # Simulamos dos objetos en la misma posición (theta, r)
    points = [
        {"name": "Venus", "theta": 0.5, "r": 40, "mag": -4.0},
        {"name": "Marte", "theta": 0.51, "r": 40.1, "mag": 1.0}
    ]

    class MockAx:
        class transData:
            def transform(coord): return (coord[0] * 100, coord[1] * 100)

    # Con cluster_px grande, debería elegir solo el más brillante (Venus)
    selected = _select_labels(MockAx, points, "inteligentes", max_labels=5, cluster_px=50)

    assert len(selected) == 1
    assert selected[0]["name"] == "Venus"