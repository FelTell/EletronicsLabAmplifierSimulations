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


def GetAdAndDiffInputRes(symbol) -> tuple[float, float, float, float]:
    rr = RunLtSpiceSimulation("Sims/0_Ad.asc", symbol)[0]
    asc = AscEditor("Sims/0_Ad.asc")
    vout = rr.get_trace("v(OUT)")
    vinNeg = rr.get_trace("v(In-)")
    vinPos = rr.get_trace("v(In+)")
    iin = rr.get_trace("Ix(x1:in-)")
    freq = rr.get_trace("frequency")
    idx = abs(freq.data - 1000).argmin()
    R2 = asc.get_component_floatvalue("R2")
    R3 = asc.get_component_floatvalue("R3")
    beta = R2 / (R2 + R3)
    gain = vout.data[idx] / (vinNeg.data[idx] - vinPos.data[idx])
    diffInputResClosed = vinNeg.data[idx] / iin.data[idx]
    diffInputResOpen = diffInputResClosed / (1 + gain * beta)
    return abs(gain), abs(diffInputResClosed), abs(diffInputResOpen), beta


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


def GetResults(symbol):
    print(f"Running simulation for symbol: {symbol}")
    ad, diffInputResClosed, diffInputResOpen, beta = GetAdAndDiffInputRes(
        symbol)
    offset, thd = GetOffsetAndThd(symbol)
    psrrPos = GetPsrrPos(symbol)
    psrrNeg = GetPsrrNeg(symbol)
    outputResClosed, outputResOpen = GetOutputRes(symbol, ad, beta)

    print(f"    Ganho em malha aberta = {EngNumber(ad)}V/V")
    print(f"    Tensão de Offset = {EngNumber(offset)}V")
    print(f"    Taxa de distorção harmônica em 1kHz = {thd:.4f}%")
    print(f"    PSRR (Positivo) = {EngNumber(psrrPos)}dB")
    print(f"    PSRR (Negativo) = {EngNumber(psrrNeg)}dB")
    print(
        f"    Resistência entrada dif. malha fechada = {EngNumber(diffInputResClosed)}Ω")
    print(
        f"    Resistência entrada dif. malha aberta = {EngNumber(diffInputResOpen)}Ω")
    print(
        f"    Resistência de saída malha fechada = {EngNumber(outputResClosed)}Ω")
    print(
        f"    Resistência de saída malha aberta = {EngNumber(outputResOpen)}Ω")
    print("")


GetResults("A_AmpBasico")
GetResults("B_CargaAtiva")
GetResults("C_Fonte1")
GetResults("D_Fonte2")
GetResults("E_Fonte2Darlington")
