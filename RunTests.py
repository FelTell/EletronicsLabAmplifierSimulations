from PyLTSpice import SimRunner, AscEditor, RawRead
from engineering_notation import EngNumber
from pathlib import Path
import sys
import numpy


def RunLtSpiceSimulation(ascPath, symbol) -> tuple[RawRead, str]:
    outPath = Path("Sims/Results")
    outPath.mkdir(exist_ok=True)
    runner = SimRunner(output_folder=str(outPath))
    asc = AscEditor(str(ascPath))
    asc.get_component("X1").symbol = symbol
    task = runner.run(asc)
    if task is None:
        print("ERROR: Simulation failed to start.")
        sys.exit(3)
    rawFile, logFile = task.wait_results()
    return RawRead(rawFile), logFile


def GetAdAndDiffInputRes(
        symbol) -> tuple[float, float, float, float, float, float]:
    rr = RunLtSpiceSimulation("Sims/0_Ad.asc", symbol)[0]
    asc = AscEditor("Sims/0_Ad.asc")
    vout = rr.get_trace("v(OUT)")
    vinNeg = rr.get_trace("v(In-)")
    vinPos = rr.get_trace("v(In+)")
    vDataOpen = abs(vout.data / (vinNeg.data - vinPos.data))
    vDataClosed = abs(vout.data / (vinPos.data))
    iin = rr.get_trace("Ix(x1:in-)")
    freq = rr.get_trace("frequency")
    idx = abs(freq.data - 1000).argmin()
    R2 = asc.get_component_floatvalue("R2")
    R3 = asc.get_component_floatvalue("R3")

    beta = R2 / (R2 + R3)
    gainOpen = vDataOpen[idx]
    diffInputResClosed = vinNeg.data[idx] / iin.data[idx]
    diffInputResOpen = diffInputResClosed / (1 + gainOpen * beta)

    gainOpen3db = 20 * numpy.log10(gainOpen) - 3
    voutBw = 10 ** (gainOpen3db / 20)
    idx3db = 0
    for i in range(len(vDataOpen)):
        if vDataOpen[i] <= voutBw:
            idx3db = i
            break
    bwOpen = freq.data[idx3db]

    gainClosed = vDataClosed[idx]
    gainClosed3db = 20 * numpy.log10(gainClosed) - 3
    voutClosedBw = 10 ** (gainClosed3db / 20)
    idx3dbClosed = 0
    for i in range(len(vDataClosed)):
        if vDataClosed[i] <= voutClosedBw:
            idx3dbClosed = i
            break
    bwClosed = freq.data[idx3dbClosed]

    return abs(gainOpen), abs(diffInputResClosed), abs(
        diffInputResOpen), beta, abs(bwOpen), abs(bwClosed)


def GetOffsetAndThd(symbol) -> tuple[float, float]:
    logFile = RunLtSpiceSimulation("Sims/1_OffsetThd.asc", symbol)[1]
    offset = 0.0
    thd = 0.0
    with open(logFile, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if 'DC component:' in line:
                offsetStr = line.split('DC component:')[1].strip()
                offset = float(offsetStr)
            elif 'Total Harmonic Distortion' in line:
                thdStr = line.split('Total Harmonic Distortion:')[1]
                thdStr = thdStr.strip().split('%')[0].strip()
                thd = float(thdStr)
    return offset, thd


def GetPsrrPos(symbol) -> float:
    rr = RunLtSpiceSimulation("Sims/2_PsrrPos.asc", symbol)[0]
    vout = rr.get_trace("v(OUT)")

    vccPos = rr.get_trace("v(V+)")
    freq = rr.get_trace("frequency")
    idx = abs(freq.data - 1000).argmin()
    psrrPos = vout.data[idx] / vccPos.data[idx]
    psrrPos_db = 20 * numpy.log10(abs(psrrPos))
    return psrrPos_db


def GetPsrrNeg(symbol) -> float:
    rr = RunLtSpiceSimulation("Sims/3_PsrrNeg.asc", symbol)[0]
    vout = rr.get_trace("v(OUT)")

    vccNeg = rr.get_trace("v(V-)")
    freq = rr.get_trace("frequency")
    idx = abs(freq.data - 1000).argmin()
    psrrNeg = vout.data[idx] / vccNeg.data[idx]
    psrrNeg_db = 20 * numpy.log10(abs(psrrNeg))
    return psrrNeg_db


def GetOutputRes(symbol, ad, beta) -> tuple[float, float]:
    rr = RunLtSpiceSimulation("Sims/5_OutputResistance.asc", symbol)[0]
    vout = rr.get_trace("v(OUT)")
    ir4 = rr.get_trace("I(R4)")
    freq = rr.get_trace("frequency")
    idx = abs(freq.data - 1000).argmin()
    outputResClosed = vout.data[idx] / ir4.data[idx]
    outputResOpen = outputResClosed * (1 + ad * beta)
    return abs(outputResClosed), abs(outputResOpen)


def GetSlewRateAndSaturation(symbol) -> tuple[float, float, float]:
    rr = RunLtSpiceSimulation("Sims/6_SlewRateSat.asc", symbol)[0]
    vout = rr.get_trace("v(OUT)")
    time = rr.get_trace("time")
    # Get saturation voltages from the middle point of the square wave
    idxSatPos = abs(time.data - 125e-06).argmin()
    idxSatNeg = abs(time.data - 375e-06).argmin()
    vSatPos = vout.data[idxSatPos]
    vSatNeg = vout.data[idxSatNeg]
    # Get start value by finding the first point where the signal crosses 90%
    # of the negative saturation voltage
    idxStartSR = 0
    idxEndSR = 0
    for i, v in enumerate(vout.data):
        if v >= vSatNeg * 0.9:
            idxStartSR = i
            break
    # Get end value by finding the last point before the signal crosses 90%
    # of the positive saturation voltage
    for i, v in enumerate(vout.data):
        if v >= vSatPos * 0.9:
            idxEndSR = i
            break
    deltaV = vout.data[idxEndSR] - vout.data[idxStartSR]
    deltaT = time.data[idxEndSR] - time.data[idxStartSR]
    slewRate = (deltaV / deltaT) * 1e-06
    return abs(slewRate), abs(float(vSatPos)), float(vSatNeg)


def GetClosedLoopGain(symbol, ad) -> float:
    rr = RunLtSpiceSimulation("Sims/7_GanhoFechada.asc", symbol)[0]
    asc = AscEditor("Sims/0_Ad.asc")
    R2 = asc.get_component_floatvalue("R2")
    R3 = asc.get_component_floatvalue("R3")
    vout = rr.get_trace("v(OUT)")
    freq = rr.get_trace("frequency")
    idx = abs(freq.data - 1000).argmin()
    acmmf = vout.data[idx]

    cmmr_linear = (2 * (1 + ad * R2) * acmmf) / (2 * R3 + R2 * acmmf)
    cmmr_db = 20 * numpy.log10(cmmr_linear)

    return abs(cmmr_db)


def GetResults(symbol):
    print(f"Running simulation for symbol: {symbol}")
    ad, diffInputResClosed, diffInputResOpen, beta, bwOpen, bwClosed = GetAdAndDiffInputRes(
        symbol)
    offset, thd = GetOffsetAndThd(symbol)
    psrrPos = GetPsrrPos(symbol)
    psrrNeg = GetPsrrNeg(symbol)
    outputResClosed, outputResOpen = GetOutputRes(symbol, ad, beta)
    slewRate, vSatPos, vSatNeg = GetSlewRateAndSaturation(symbol)
    cmmr = GetClosedLoopGain(symbol, ad)

    print(f"    CMRR = {EngNumber(cmmr)}dB")
    print(f"    PSRR (Positivo) = {EngNumber(psrrPos)}dB")
    print(f"    PSRR (Negativo) = {EngNumber(psrrNeg)}dB")
    print(f"    Ganho em malha aberta = {EngNumber(ad)}V/V")
    print(f"    Banda passante em malha aberta = {EngNumber(bwOpen)}Hz")
    print(f"    Banda passante em malha fechada = {EngNumber(bwClosed)}Hz")
    print(f"    Res. entrada malha fechada = {EngNumber(diffInputResClosed)}Ω")
    print(f"    Tensão de Saturação (Positiva) = {EngNumber(vSatPos)}V")
    print(f"    Tensão de Saturação (Negativa) = {EngNumber(vSatNeg)}V")
    print(f"    Tensão de Offset = {EngNumber(offset)}V")
    print(f"    Slew Rate = {EngNumber(slewRate)}V/μs")
    print(f"    Taxa de distorção harmônica em 1kHz = {thd:.4f}%")
    print(f"    Res. entrada malha fechada = {EngNumber(diffInputResClosed)}Ω")
    print(f"    Res. entrada malha aberta = {EngNumber(diffInputResOpen)}Ω")
    print(f"    Res. saída malha fechada = {EngNumber(outputResClosed)}Ω")
    print(f"    Res. saída malha aberta = {EngNumber(outputResOpen)}Ω")
    print("")


GetResults("A_AmpBasico")
GetResults("B_CargaAtiva")
GetResults("C_Fonte1")
GetResults("D_Fonte2")
GetResults("E_Fonte2Darlington")
GetResults("F_FonteCorrenteAmp")
GetResults("G_CC-CE")
GetResults("H_Cascode")
GetResults("I_FonteComum")
GetResults("J_SeguidorEmissor")
GetResults("K_ParComplementar")
GetResults("L_Tripple")

GetResults("Z_Final")


# banda passante 4hz -> 210khz
# Tsat = 7.4v
