# Wenglor-Kreisbahn-Simulator und Mehrprofil-Auswertung

## Dateien

- `simulate_profile_raycast.py`: erzeugt eine gebogene Blechkontur und simuliert Profile entlang einer Kreisbahn. Standardmäßig beträgt der Bahnradius 200 mm, der Scanbogen wird um 30° im Uhrzeigersinn gedreht und nur die von der ersten Scanpose sichtbare Blechfläche bleibt aktiv. X/Z werden in jeder Zeile im lokalen Sensor-Koordinatensystem gespeichert.
- `bend_measure_multiview.py`: wertet lokale Einzelprofile separat aus und fasst gültige Winkel- und Radiuswerte statistisch zusammen. Für die Auswertung werden keine Sensorwinkel benötigt.
Voraussetzung: Python 3 und NumPy; für PNG-Visualisierungen zusätzlich Matplotlib (`python3 -m pip install numpy matplotlib`). Die Auswertung speichert standardmäßig neben der CSV eine Datei `*_auswertung.png`; mit `--no-plot` lässt sich das abschalten und mit `--plot-path` ein anderer Speicherort festlegen. Dieses Paket enthält keine Beispieldatensätze.

## Testlauf

```bash
python3 simulate_profile_raycast.py --orbit-center-x 0 --orbit-center-z 0 --orbit-radius 200 --scan-rotation-deg -30 --bend-deg 20 --inner-radius 3 --thickness 2 --noise 0.05 --intensity-noise 5 --seed 1234 --no-metadata --no-png
python3 bend_measure_multiview.py sim_profile/<erzeugte-csv-datei>.csv
```

Der Basis-Scan läuft von −60° bis +60° in 5°-Schritten (25 Profile); die Standardrotation −30° ergibt absolute Sensorwinkel von −90° bis +30°. Die 2048 Strahlen je Profil und alle Parameter im Kommando sind illustrative Testwerte, keine kalibrierten Anlagenwerte.

## Getrennter Sensor-Drehpunkt und Biegezentrum

Der Sensor-Drehpunkt bleibt fest; der Sensor bewegt sich auf einer Kreisbahn darum. Das Biegezentrum des Werkstücks ist davon getrennt. Sein Abstand zum nominalen Biegezentrum kann über `--pivot-offset-x` und `--pivot-offset-z` eingestellt werden. Die Standardwerte 0 mm sind nur ein neutraler Platzhalter und bilden den beschriebenen Zentrenversatz nicht ab. Bis die reale Lage gemessen ist, müssen für einen realitätsnahen Test passende Beispielwerte angegeben werden.

Illustrativer Pivotversatz sowie eine Blechlage um 10 mm oberhalb der Mitte des verfügbaren Verschiebewegs:

```bash
python3 simulate_profile_raycast.py \
  --pivot-offset-x 30 --pivot-offset-z 20 \
  --bend-shift-z 10 --no-png
```

Die Offset-Werte sind **keine Messwerte der Anlage**. `--orbit-center-x` und `--orbit-center-z` erlauben alternativ, die absoluten Koordinaten des festen Sensor-Drehpunkts vorzugeben.

## Verschiebbarer Blech-/Biegebereich

Der Simulator verwendet standardmäßig einen Gesamtweg von 50 mm entlang Z. `--bend-shift-z` gibt die Lage relativ zur Wegmitte an; bei 50 mm Gesamtweg ist der zulässige Bereich daher −25 mm bis +25 mm. Der Sensor-Drehpunkt bleibt beim Verschieben unverändert, nur das Werkstück wird translatiert. Der Weg und die Verschiebung sind über `--max-bend-travel` bzw. `--bend-shift-z` einstellbar.

## Sensorbahn und Profilpose

- `--scan-start-deg`, `--scan-end-deg`, `--scan-step-deg`: Scanbogen und Winkelschritt; Winkel werden von +X gegen den Uhrzeigersinn gemessen.
- `--orbit-radius`: Bahnradius um den festen Sensor-Drehpunkt.
- `--fov-deg`, `--rays`: Sichtfeld und Strahlenzahl.
- `--scan-rotation-deg`: Drehung des gesamten Scanbogens; negative Werte drehen im Uhrzeigersinn.
- Standardmäßig wird die von der ersten Sensorpose sichtbare Blechfläche festgelegt. Treffer auf der gegenüberliegenden Fläche werden verworfen; `--both-sides` hebt die Sperre auf.
- X und Z sind immer lokale Koordinaten des jeweiligen Sensorprofils. Alle Profile stehen zeilenweise in einer CSV; `Profile` kennzeichnet die jeweilige Gruppe. Damit werden die Datensätze in einer Datei gesammelt, aber nicht geometrisch zu einer globalen Punktwolke registriert.
- Standardmäßig enthält das CSV zusätzlich Kommentar-Metadaten und Sensorpose-Spalten. Mit `--no-metadata` entfallen sie; `X [mm],Z [mm],Intensity,Profile` bleiben erhalten. Für die Winkel- und Radiusaggregation wird kein Sensorwinkel benötigt.

Für die reale Anlage wurde angegeben, dass die Profilpose nicht gespeichert wird und der Scan nicht gleichmäßig mit reproduzierbarer Profilanzahl abläuft. Deshalb wird kein Winkel aus der Profilnummer abgeleitet. Die einzelnen lokalen Profile werden in einer Datei gesammelt und einzeln ausgewertet; ihre Messergebnisse können aggregiert werden. Eine gemeinsame globale Punktwolke lässt sich ohne Sensorpose nicht rekonstruieren.

## Mehrprofil-Auswertung und Mittelpunkt

Die Auswertung behandelt X/Z als lokale Koordinaten des jeweiligen Profils. Sie entfernt erkannte Gerade der Schenkel und sucht in den verbleibenden Punkten robust nach einem Kreisbogen. Biegezentrum und sichtbarer Radius können so pro Profil geschätzt werden, sofern genügend Bogenpunkte vorliegen. Gültige Winkel und Radien werden danach statistisch zusammengefasst. Eine globale X/Z-Lage oder geometrisch registrierte Punktwolke lässt sich ohne Sensorpose nicht bestimmen. Werden nur Punkte einer Bogenfläche erkannt, wird der sichtbare Radius ausgegeben, aber nicht sicher als Innen- oder Außenradius bezeichnet.

Ohne ausreichend gekrümmte Bogenpunkte kann weiterhin nur der Winkel aus den Schenkellinien bestimmt werden. Der Sensor-Drehpunkt ist niemals ein Ersatz für das Biegezentrum.

Die Profile werden als lokale Einzelprofile in einer CSV gesammelt. Bei echten Sensordaten muss der Export jedem Profil eine fortlaufende Profilnummer zuweisen; diese Nummer ist nur eine Gruppierungskennung und kein Sensorwinkel. Jedes Profil, in dem beide Schenkel ausreichend sichtbar sind, liefert einen Öffnungswinkel; geeignete Profile liefern außerdem einen Radius. Die gültigen Einzelwerte werden per Median zusammengefasst und ihre Streuung ausgegeben. Das ist eine Zusammenführung der Messergebnisse, keine geometrische Registrierung der Punkte. Zeigt kein einzelnes Profil beide Schenkel, lassen sich Teilprofile ohne Sensorpose nicht sicher zu einer gemeinsamen Geometrie zusammensetzen.

Die kleinste unterstützte Öffnung ist standardmäßig 30° (`--min-opening`).
## Grenzen des Modells

Das Modell ist ein 2D-Raytracing mit zentralem Sensorpunkt und erstem Treffer je Strahl. Es bildet die weCat3D-Triangulationsoptik, Reflexionen, Materialeffekte, Halterungen und reale Ausfälle nicht kalibriert nach. Parameterwerte sind nicht als Anlagenmesswerte zu verstehen.
