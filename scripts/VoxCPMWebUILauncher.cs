using System;
using System.Diagnostics;
using System.IO;
using System.Text;

public static class VoxCPMWebUILauncher
{
    public static int Main(string[] args)
    {
        string repoRoot = AppDomain.CurrentDomain.BaseDirectory;
        string scriptPath = Path.Combine(repoRoot, "scripts", "launch_webui.ps1");

        if (!File.Exists(scriptPath))
        {
            Console.Error.WriteLine("Missing launcher script: " + scriptPath);
            return 1;
        }

        var psArgs = new StringBuilder();
        psArgs.Append("-NoProfile -ExecutionPolicy Bypass -File ");
        psArgs.Append(Quote(scriptPath));

        foreach (string arg in args)
        {
            psArgs.Append(" ");
            psArgs.Append(Quote(arg));
        }

        var startInfo = new ProcessStartInfo
        {
            FileName = "powershell.exe",
            Arguments = psArgs.ToString(),
            WorkingDirectory = repoRoot,
            UseShellExecute = false,
        };

        try
        {
            using (Process process = Process.Start(startInfo))
            {
                process.WaitForExit();
                return process.ExitCode;
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine("Failed to start VoxCPM Web UI: " + ex.Message);
            return 1;
        }
    }

    private static string Quote(string value)
    {
        if (value == null)
        {
            return "\"\"";
        }

        return "\"" + value.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
    }
}
