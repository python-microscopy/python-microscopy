pa.stop()
import piezo_e662
pfoc = piezo_e662.piezo_e662()
pfoc.initialise()

import simplesequenceaquisator
sa = simplesequenceaquisator.SimpleSequenceAquisitor(chaninfo, cam, pfoc)
sa.SetStartMode(sa.START_AND_END)
sa.SetStepSize(2)
sa.SetStartPos(50)
sa.SetEndPos(70)
sa.Prepare()
sa.start()