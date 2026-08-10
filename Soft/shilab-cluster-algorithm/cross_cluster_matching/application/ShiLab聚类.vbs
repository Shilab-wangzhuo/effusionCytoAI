Set oShell = CreateObject("WScript.Shell")
oShell.Run "cmd.exe /c conda activate learn && pythonw " & _
    Chr(34) & "E:\limr\Soft\shilab-cluster-algorithm\cross_cluster_matching\application\pipeline_gui.py" & Chr(34), _
    0, False
