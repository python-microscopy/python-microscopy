#minimal protocol which does nothing
class Protocol:
    def __init__(self):
        pass

    def Init(self):
        pass
    
    def OnFrame(self, frameNum):
        pass

NullProtocol = Protocol()

class TaskListTask:
    def __init__(self, when, what, *withParams):
        self.when = when
        self.what = what
        self.params = withParams

T = TaskListTask #to save typing in protocols

class TaskListProtocol(Protocol):
    def __init__(self, taskList):
        self.taskList = taskList
        Protocol.__init__(self)
        self.listPos = 0

    def Init(self):
        self.listPos = 0

        self.OnFrame(-1) #do everything which needs to be done before acquisition starts

    def OnFrame(self, frameNum):
        while not self.listPos >= len(self.taskList) and frameNum >= self.taskList[self.listPos].when:
            t = self.taskList[self.listPos]
            t.what(*t.params)
            self.listPos += 1

def Ex(str):
    exec(str)
