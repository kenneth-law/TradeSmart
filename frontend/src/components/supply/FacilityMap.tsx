import { useEffect, useRef } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import type { SupplyChainFacility, SupplyChainFacilityType } from '../../types'

export const FACILITY_STYLE: Record<SupplyChainFacilityType, { color: string; label: string }> = {
  manufacturing: { color: '#F59E0B', label: 'Manufacturing' },
  distribution: { color: '#22C55E', label: 'Distribution' },
  warehouse: { color: '#06B6D4', label: 'Warehouse' },
  research: { color: '#A855F7', label: 'R&D' },
  office: { color: '#60A5FA', label: 'Office' },
  headquarters: { color: '#EF4444', label: 'Headquarters' },
  other: { color: '#94A3B8', label: 'Other' },
}

export default function FacilityMap({
  facilities,
  selectedId,
  onSelect,
}: {
  facilities: SupplyChainFacility[]
  selectedId: string | null
  onSelect: (id: string) => void
}) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const mapRef = useRef<L.Map | null>(null)
  const layerRef = useRef<L.LayerGroup | null>(null)
  const onSelectRef = useRef(onSelect)
  onSelectRef.current = onSelect

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return
    const map = L.map(containerRef.current, {
      center: [20, 0],
      zoom: 2,
      minZoom: 2,
      maxZoom: 17,
      worldCopyJump: true,
      zoomControl: true,
    })
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors',
      maxZoom: 19,
    }).addTo(map)
    layerRef.current = L.layerGroup().addTo(map)
    mapRef.current = map

    const observer = new ResizeObserver(() => map.invalidateSize({ animate: false }))
    observer.observe(containerRef.current)
    window.setTimeout(() => map.invalidateSize({ animate: false }), 0)
    return () => {
      observer.disconnect()
      map.remove()
      mapRef.current = null
      layerRef.current = null
    }
  }, [])

  useEffect(() => {
    const map = mapRef.current
    const layer = layerRef.current
    if (!map || !layer) return
    layer.clearLayers()
    const bounds: L.LatLngExpression[] = []

    facilities.forEach(facility => {
      const style = FACILITY_STYLE[facility.facility_type]
      const selected = facility.id === selectedId
      const latLng: L.LatLngExpression = [facility.latitude, facility.longitude]
      bounds.push(latLng)
      const marker = L.circleMarker(latLng, {
        radius: selected ? 9 : facility.facility_type === 'headquarters' ? 7 : 5.5,
        color: selected ? '#FFFFFF' : style.color,
        weight: selected ? 2.5 : 1.5,
        fillColor: style.color,
        fillOpacity: selected ? 1 : 0.82,
      })
      marker.bindTooltip(
        `<strong>${escapeHtml(facility.facility_name)}</strong><br>${escapeHtml(facility.company_name)} · ${escapeHtml(style.label)}`,
        { direction: 'top', offset: [0, -6] },
      )
      marker.on('click', () => onSelectRef.current(facility.id))
      marker.addTo(layer)
    })

    if (bounds.length === 1) map.setView(bounds[0], 7)
    else if (bounds.length > 1) map.fitBounds(L.latLngBounds(bounds), { padding: [36, 36], maxZoom: 7 })
  }, [facilities, selectedId])

  return <div ref={containerRef} className="facility-map relative isolate z-0 h-full min-h-[460px] w-full bg-[#071019]" aria-label="Company facilities world map" />
}

function escapeHtml(value: string) {
  return value.replace(/[&<>'"]/g, character => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    "'": '&#39;',
    '"': '&quot;',
  })[character] ?? character)
}
