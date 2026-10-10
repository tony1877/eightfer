// shoehorn-hwmon: reads the temperature and voltage sensors through LibreHardwareMonitorLib every 2 s and serves them as
// /data.json on 127.0.0.1 (port 8085, or the first argument), in the tree shape of LibreHardwareMonitor's own web
// server: root > computer > hardware > sensors, each sensor with SensorId, Type, Text and Value. Sub-hardware sensors
// are listed under their hardware. Only temperatures and voltages are served; nothing listens beyond this machine.

using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Runtime.InteropServices;
using System.Text.Json;
using LibreHardwareMonitor.Hardware;

// A driver update or reset can fault inside the sensor libraries (NVML), which .NET cannot catch. So this process is
// a supervisor: it runs itself with --worker as a child and starts it again when it exits. No crash dialog for either
// (the error mode is inherited).
SetErrorMode(0x0001 | 0x0002);  // SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX
if (!args.Contains("--worker"))
{
    string self = Environment.ProcessPath!;
    while (true)
    {
        var psi = new ProcessStartInfo(self) { UseShellExecute = false };
        psi.ArgumentList.Add("--worker");
        foreach (string a in args) psi.ArgumentList.Add(a);
        using var child = Process.Start(psi);
        child?.WaitForExit();
        Thread.Sleep(3000);
    }
}

string? portArg = args.FirstOrDefault(a => a != "--worker");
int port = portArg != null && int.TryParse(portArg, out var p) ? p : 8085;

var computer = new Computer
{
    IsCpuEnabled = true,
    IsGpuEnabled = true,
    IsMemoryEnabled = true,
    IsMotherboardEnabled = true,
    IsStorageEnabled = true,
    IsControllerEnabled = true,
};
computer.Open();

// Board voltage maps LibreHardwareMonitorLib lacks: sensor name -> (new name, factor on the library's value; 0 = drop).
// ROG CROSSHAIR X670E GENE (NCT6799D): the library lists the inputs raw ("This is wrong" on Vcore and VTT). The
// dividers of ASUS's ROG CROSSHAIR X870E HERO / APEX map (same input layout, matched to HWiNFO in the library) fit
// it: the 1.8 V PLL reads 1.80 V and both chipsets their nominal 1.05 V.
var boardMaps = new Dictionary<string, Dictionary<string, (string Name, double Factor)>>
{
    ["ROG CROSSHAIR X670E GENE"] = new()
    {
        ["Vcore"] = ("Vcore", 151.0 / 136 / 2),  // the library already doubled it
        ["Voltage #6"] = ("CPU VDD_MISC", 91.0 / 82),
        ["Voltage #7"] = ("CPU SoC", 91.0 / 82),
        ["CPU Termination"] = ("", 0),
        ["Voltage #11"] = ("Chipset 1 VDD", 2),
        ["Voltage #12"] = ("Chipset 2 VDD", 2),
        ["Voltage #13"] = ("Chipset Standby", 1),
        ["Voltage #14"] = ("CPU VDDIO Memory", 91.0 / 82),
        ["Voltage #15"] = ("1.8V PLL", 53.0 / 36),
    },
};

byte[] body = "{}"u8.ToArray();
var gate = new object();

void Sample()
{
    var hardware = new List<object>();
    foreach (IHardware hw in computer.Hardware)
    {
        hw.Update();
        var sensors = new List<object>();
        void Add(IHardware h)
        {
            foreach (ISensor s in h.Sensors)
            {
                if (s.SensorType is not (SensorType.Temperature or SensorType.Voltage) || s.Value is not float v) continue;
                bool volt = s.SensorType == SensorType.Voltage;
                string name = s.Name;
                var map = volt ? boardMaps.FirstOrDefault(b => Clean(hw.Name).Contains(b.Key, StringComparison.OrdinalIgnoreCase)).Value : null;
                if (map != null && map.TryGetValue(name, out var m))
                {
                    if (m.Factor == 0) continue;
                    name = m.Name;
                    v = (float) (v * m.Factor);
                }
                sensors.Add(new Dictionary<string, object>
                {
                    ["Text"] = name,
                    ["SensorId"] = s.Identifier.ToString(),
                    ["Type"] = volt ? "Voltage" : "Temperature",
                    ["Value"] = v.ToString(volt ? "0.000" : "0.0", CultureInfo.InvariantCulture) + (volt ? " V" : " C"),
                });
            }
        }
        Add(hw);
        foreach (IHardware sub in hw.SubHardware)
        {
            sub.Update();
            Add(sub);
        }
        hardware.Add(new Dictionary<string, object> { ["Text"] = Clean(hw.Name), ["Children"] = sensors });
    }
    var root = new Dictionary<string, object>
    {
        ["Text"] = "Sensor",
        ["Children"] = new[] { new Dictionary<string, object> { ["Text"] = Environment.MachineName, ["Children"] = hardware } },
    };
    byte[] json = JsonSerializer.SerializeToUtf8Bytes(root);
    lock (gate) body = json;
}

// names read from hardware (a memory module's SPD part number) can carry unprintable bytes
static string Clean(string s) =>
    System.Text.RegularExpressions.Regex.Replace(new string(s.Where(c => c >= ' ' && c <= '~').ToArray()), " {2,}", " ").Trim();

Sample();
var timer = new Timer(_ => { try { Sample(); } catch { } }, null, 2000, 2000);

var listener = new HttpListener();
listener.Prefixes.Add($"http://127.0.0.1:{port}/");
listener.Start();
while (true)
{
    HttpListenerContext ctx = listener.GetContext();
    try
    {
        byte[] b;
        lock (gate) b = body;
        if (ctx.Request.Url?.AbsolutePath == "/data.json")
        {
            ctx.Response.ContentType = "application/json";
            ctx.Response.OutputStream.Write(b);
        }
        else
        {
            ctx.Response.StatusCode = 404;
        }
    }
    catch { }
    finally { ctx.Response.Close(); }
}

[DllImport("kernel32.dll")]
static extern uint SetErrorMode(uint mode);
