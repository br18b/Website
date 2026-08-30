---
title: "Saab 92a"
permalink: /BNG/S92a/
sort_order: 20
category: "BeamNG project"
status: "Work in progress"
status_class: "wip"
summary: "A 1949–1957 Swedish two-stroke built by airplane people, shaped like a tiny aerodynamic soap bubble."
images:
  - src: /BNG/pics/Saab92_1.png
    alt: Front view
  - src: /BNG/pics/Saab92_2.png
    alt: Rear view
  - src: /BNG/pics/Saab92_3.png
    alt: Interior
  - src: /BNG/pics/Saab92_4.png
    alt: Engine bay
cover_images:
  - src: /BNG/pics/Saab92_1.png
    alt: Front view
  - src: /BNG/pics/Saab92_2.png
    alt: Rear view
  - src: /BNG/pics/Saab92_3.png
    alt: Dashboard view
  - src: /BNG/pics/Saab92_4.png
    alt: Engine view
stats:
  Top speed: "~100 km/h (60 mph)"
  Weight: "~800 kg"
  Engine: "water cooled 764cc 2-stroke I2"
  Transmission: "3-speed, freewheel"
  Torque: "88 Nm (65 ft.lbs) @ 3000 rpm"
  Power: "50 hp (35 kW) @ 4600 rpm"
  Acceleration 0-60: "Maybe"
  Variants: "2-door passenger"
---

## About the car

{% include figure.html src="/BNG/pics/Saab92_1.png" alt="Saab 92" caption="Default factory version in its iconic green color, 1949." align="right" %}

Ever seen a car made by an airplane company? This is it. The Saab 92A was Saab’s very first production car, and it shows in the best possible way. Born in post-war Sweden and shaped by the same engineers who built fighter planes, it had a drag coefficient of just 0.3, making it one of the most aerodynamic cars of its time.

Do not let the slippery shape fool you, though. This thing was not breaking speed records. Under the hood was a tiny two-stroke, 760cc engine that sounded more like a chainsaw — or maybe an especially angry lawnmower — than a car. It buzzed and shrieked like a swarm of bees barreling toward you at full throttle.

To get there, you had to keep the revs screaming in that glorious two-stroke comfort zone: somewhere between “this is fine” and “this engine might just lift off.” Fitting, really, for something designed by people who used to build airplanes.

## About the mod

As of now, this mod features:
- fully openable doors and hood
- togglable interior knobs:
  - interior lights, headlights
  - windshield wipers cycles
  - choke
  - ignition
  - rooftop lights (cab & police version only)
- cold engine start behavior: custom lua script that prevents the car from starting if the engine temperature is too cold (can be turned off)
- jbeam-based animated windshield wipers
- freewheeling — the engine RPM will fall down even in gear if the gas is not pressed — will act as if in neutral/clutch is pressed (can be removed in the parts selector). The arcade shifting logic was fully adjusted to acommodate this behavior without any hiccups. Upshifting can now be done clutchless to mimick the real-world behavior correctly.