# shoehorn-hwmon

A windowless helper that reads the temperature and voltage sensors through
[LibreHardwareMonitorLib](https://github.com/LibreHardwareMonitor/LibreHardwareMonitor) (MPL-2.0) every 2 s and
serves them as `/data.json` on `127.0.0.1:8085`, the format of LibreHardwareMonitor's own web server, so
`shoehorn serve --hw-monitor` reads either. It needs administrator rights (CPU, memory module and board sensors go
through a kernel driver) and listens on this machine only.

```powershell
dotnet publish tools\hwmon -c Release -o C:\Tools\shoehorn-hwmon   # .NET 8 SDK
C:\Tools\shoehorn-hwmon\shoehorn-hwmon.exe [port]                 # asks for elevation
```

Start at logon without a prompt (run once in an elevated PowerShell):

```powershell
Register-ScheduledTask shoehorn-hwmon -RunLevel Highest -Trigger (New-ScheduledTaskTrigger -AtLogOn) `
    -Action (New-ScheduledTaskAction -Execute C:\Tools\shoehorn-hwmon\shoehorn-hwmon.exe) `
    -Settings (New-ScheduledTaskSettingsSet -ExecutionTimeLimit 0 -AllowStartIfOnBatteries)
```
