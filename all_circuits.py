import schemdraw
import schemdraw.elements as elm

print("Generating 4.5.1 Inverting Amplifier...")
with schemdraw.Drawing(file='4_5_1_Inverting_Amplifier.svg') as d1:
    op = d1.add(elm.Opamp(sign=True))
    
    # Inverting input path (Pin 2)
    d1.add(elm.Line(d='left', xy=op.in1, l=d1.unit/4))
    d1.add(elm.Resistor(d='left', label='R1\n10kΩ'))
    d1.add(elm.SourceV(d='down', label='V1'))
    d1.add(elm.Ground())
    
    # Non-inverting input path (Pin 3)
    d1.add(elm.Line(d='left', xy=op.in2, l=d1.unit/4))
    d1.add(elm.Ground())
    
    # Feedback loop
    d1.add(elm.Line(d='up', xy=op.in1, l=d1.unit/2))
    d1.add(elm.Resistor(d='right', label='R2\n22kΩ', tox=op.out))
    d1.add(elm.Line(d='down', toy=op.out))
    
    # Output (Pin 6)
    d1.add(elm.Line(d='right', xy=op.out, l=d1.unit/4))
    d1.add(elm.Dot())
    d1.add(elm.Line(d='right', l=d1.unit/4, label='Vout'))

print("Generating 4.5.2 Non-Inverting Amplifier...")
with schemdraw.Drawing(file='4_5_2_NonInverting_Amplifier.svg') as d2:
    op = d2.add(elm.Opamp(sign=True))
    
    # Inverting input path
    d2.add(elm.Line(d='left', xy=op.in1, l=d2.unit/4))
    d2.add(elm.Dot())
    d2.push()
    d2.add(elm.Resistor(d='down', label='R1\n10kΩ'))
    d2.add(elm.Ground())
    d2.pop()
    
    # Feedback loop
    d2.add(elm.Line(d='up', l=d2.unit/2))
    d2.add(elm.Resistor(d='right', label='R2\n22kΩ', tox=op.out))
    d2.add(elm.Line(d='down', toy=op.out))
    
    # Non-inverting input path
    d2.add(elm.Line(d='left', xy=op.in2, l=d2.unit/4))
    d2.add(elm.SourceV(d='down', label='V1'))
    d2.add(elm.Ground())

    # Output
    d2.add(elm.Line(d='right', xy=op.out, l=d2.unit/4))
    d2.add(elm.Dot())
    d2.add(elm.Line(d='right', l=d2.unit/4, label='Vout'))

print("Generating 4.5.3 Differential Amplifier...")
with schemdraw.Drawing(file='4_5_3_Differential_Amplifier.svg') as d3:
    op = d3.add(elm.Opamp(sign=True))
    
    # Inverting input path
    d3.add(elm.Line(d='left', xy=op.in1, l=d3.unit/4))
    d3.add(elm.Dot())
    d3.push()
    d3.add(elm.Resistor(d='left', label='R1\n10kΩ'))
    d3.add(elm.SourceV(d='down', label='V1'))
    d3.add(elm.Ground())
    d3.pop()
    
    # Feedback loop
    d3.add(elm.Line(d='up', l=d3.unit/2))
    d3.add(elm.Resistor(d='right', label='R2\n22kΩ', tox=op.out))
    d3.add(elm.Line(d='down', toy=op.out))
    
    # Non-inverting input path
    d3.add(elm.Line(d='left', xy=op.in2, l=d3.unit/4))
    d3.add(elm.Dot())
    d3.push()
    d3.add(elm.Resistor(d='left', label='R3\n10kΩ'))
    d3.add(elm.SourceV(d='down', label='V2'))
    d3.add(elm.Ground())
    d3.pop()
    d3.add(elm.Resistor(d='down', label='R4\n22kΩ'))
    d3.add(elm.Ground())
    
    # Output
    d3.add(elm.Line(d='right', xy=op.out, l=d3.unit/4))
    d3.add(elm.Dot())
    d3.add(elm.Line(d='right', l=d3.unit/4, label='Vout'))

print("Generating 4.5.5 Low-Pass Filter...")
with schemdraw.Drawing(file='4_5_5_LowPass_Filter.svg') as d4:
    op = d4.add(elm.Opamp(sign=True))
    
    # Inverting input path
    d4.add(elm.Line(d='left', xy=op.in1, l=d4.unit/4))
    d4.add(elm.Dot())
    d4.push()
    d4.add(elm.Resistor(d='left', label='R1\n10kΩ'))
    d4.add(elm.SourceV(d='down', label='V1'))
    d4.add(elm.Ground())
    d4.pop()
    
    # Feedback loop (Resistor and Capacitor in parallel)
    d4.add(elm.Line(d='up', l=d4.unit/2))
    d4.add(elm.Dot())
    d4.push()
    d4.add(elm.Line(d='up', l=d4.unit/4))
    d4.add(elm.Capacitor(d='right', label='C\n0.1µF', tox=op.out))
    d4.add(elm.Line(d='down', l=d4.unit/4))
    d4.add(elm.Dot())
    d4.pop()
    d4.add(elm.Resistor(d='right', label='Rf\n10kΩ', tox=op.out))
    d4.add(elm.Line(d='down', toy=op.out))
    
    # Non-inverting input path
    d4.add(elm.Line(d='left', xy=op.in2, l=d4.unit/4))
    d4.add(elm.Ground())
    
    # Output
    d4.add(elm.Line(d='right', xy=op.out, l=d4.unit/4))
    d4.add(elm.Dot())
    d4.add(elm.Line(d='right', l=d4.unit/4, label='Vout'))

print("Done! Check your folder for the SVG images.")